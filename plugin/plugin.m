// SPDX-License-Identifier: GPL-2.0-or-later
//
// An OBS video filter that trails whatever is nearer than a cutoff distance with three delayed copies of itself:
// red, then blue, then green, each later than the one before, between the foreground and the background. Standing
// still, the copies sit hidden behind the foreground; moving, they follow it. The distance comes from Depth Pro 2,
// run with Core ML on a background queue: the picture keeps the source's size and frame rate, and the mask follows
// it at the model's.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <obs-module.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

OBS_DECLARE_MODULE()

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Trails the foreground of a source with red, blue and green delayed copies, with Depth Pro 2";
}

// Depth in meters is focal length / (width * inverse depth), and an OBS source carries no focal length.
// ponytail: one assumed lens, about 71 degrees across, so the cutoff is in approximate meters and the slider
// absorbs the error. Add a field-of-view setting if true meters ever matter.
static const float focalOverWidth = 0.7f;

// Past frames kept for the trails, at the model's input size: 1.6 s of video at 60 frames per second.
// ponytail: a fixed count, about 90 MB of video memory at 640x360 and four times that at 1280x720. The longest
// trail, three layers 500 ms apart, needs 1.5 s; size the ring from the delay if a larger model makes it too heavy.
enum { RING = 96 };

@interface TrailsFilter : NSObject {
@public
	obs_source_t *source;
	dispatch_queue_t queue; // loads and runs the model, one block at a time
	_Atomic(float) cutoff, delay, opacity, blur;
	atomic_bool busy; // a frame is with the queue

	// Graphics thread only.
	gs_effect_t *effect;
	gs_texrender_t *picture;    // the filter's input, at its own size
	gs_texrender_t *ring[RING]; // past frames at the model's size: the picture, with the foreground as alpha
	double times[RING];         // when each was stored, on `clock`
	int head, count;            // the newest frame of the ring, and how many it holds
	double clock;               // seconds of video so far
	BOOL fresh;                 // OBS has moved to a new frame since the last one was stored
	gs_stagesurf_t *stage;
	gs_texture_t *maskTexture;

	pthread_mutex_t lock; // guards the five below
	int width, height;    // the model's input size, 0 until a model is loaded
	uint8_t *input;       // RGBA, the frame the model runs on next
	_Float16 *mask;       // inverse depth of the last frame it ran on
	BOOL maskDirty;

	// Queue only.
	MLModel *model;
	MLState *state; // the earlier frames of a model with a temporal window, nil for a single-frame model
	MLMultiArray *frame;
	id<MLFeatureProvider> features;
	NSString *outputName;
	NSString *loaded; // the model and size last asked for
	NSURL *compiled;  // the temporary compiled copy of an .mlpackage
	int timed;        // predictions since the model was loaded
	uint64_t spent;   // nanoseconds in those after the first ten, which include the warm-up
}
@end

@implementation TrailsFilter

- (instancetype)init
{
	if ((self = [super init])) {
		queue = dispatch_queue_create("depth-rgb-trails", DISPATCH_QUEUE_SERIAL);
		pthread_mutex_init(&lock, NULL);
	}
	return self;
}

- (void)dealloc
{
	pthread_mutex_destroy(&lock);
	free(input);
	free(mask);
}

- (void)unload
{
	model = nil;
	state = nil;
	if (compiled)
		[NSFileManager.defaultManager removeItemAtURL:compiled error:nil];
	compiled = nil;
	pthread_mutex_lock(&lock);
	width = height = 0;
	pthread_mutex_unlock(&lock);
}

/// On the queue. `function` names one input size of a package with several, `res_<W>x<H>`; empty is its default.
- (void)load:(NSString *)path function:(NSString *)function
{
	NSString *wanted = [NSString stringWithFormat:@"%@ %@", path, function];
	if ([wanted isEqualToString:loaded])
		return;
	loaded = wanted;
	[self unload];
	// A file inside the package stands for the package, for a file dialog that opens it as a folder.
	for (NSString *p = path; p.length > 1; p = p.stringByDeletingLastPathComponent)
		if ([p.pathExtension isEqualToString:@"mlpackage"])
			path = p;
	if (!path.length)
		return;

	NSError *error;
	NSURL *url = [NSURL fileURLWithPath:path];
	if (![path.pathExtension isEqualToString:@"mlmodelc"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations" // the synchronous form: this queue is the place to wait
		url = compiled = [MLModel compileModelAtURL:url error:&error];
#pragma clang diagnostic pop
	}
	MLModelConfiguration *config = [MLModelConfiguration new];
	config.computeUnits = MLComputeUnitsCPUAndGPU; // the packages are exported for the GPU
	config.functionName = function.length ? function : nil;
	MLModel *candidate = url ? [MLModel modelWithContentsOfURL:url configuration:config error:&error] : nil;
	if (url && !candidate && function.length) { // a package exported at one size has no such function
		blog(LOG_WARNING, "[depth-rgb-trails] %s has no size %s; using its default", path.lastPathComponent.UTF8String,
		     function.UTF8String);
		config.functionName = nil;
		candidate = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
	}
	MLFeatureDescription *in = candidate.modelDescription.inputDescriptionsByName.allValues.firstObject;
	NSArray<NSNumber *> *shape = in.multiArrayConstraint.shape; // 1, 3, H, W
	if (shape.count != 4) {
		blog(LOG_ERROR, "[depth-rgb-trails] cannot load %s %s: %s", path.UTF8String, function.UTF8String,
		     error ? error.localizedDescription.UTF8String : "its input is not one RGB frame");
		[self unload];
		return;
	}
	int h = shape[2].intValue, w = shape[3].intValue;
	frame = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&error];
	features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{in.name: frame} error:&error];
	outputName = candidate.modelDescription.outputDescriptionsByName.allKeys.firstObject;
	state = candidate.modelDescription.stateDescriptionsByName.count ? [candidate newState] : nil;
	model = candidate;
	timed = 0;
	spent = 0;

	pthread_mutex_lock(&lock);
	free(input);
	free(mask);
	input = calloc((size_t)w * h, 4);
	mask = calloc((size_t)w * h, sizeof(_Float16));
	width = w;
	height = h;
	maskDirty = YES; // all zero, which is infinitely far: no foreground until the model's first answer
	pthread_mutex_unlock(&lock);
	blog(LOG_INFO, "[depth-rgb-trails] loaded %s at %dx%d, %s", path.lastPathComponent.UTF8String, w, h,
	     state ? "with a temporal window" : "single frame");
}

/// On the queue: runs the model on `input` and leaves its inverse depth in `mask`.
- (void)infer
{
	if (!model)
		return;
	int w = width, h = height, n = w * h; // only this queue changes them
	[frame getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
		_Float16 *rgb = bytes; // planar, in [0, 1]
		pthread_mutex_lock(&self->lock);
		for (int i = 0; i < n; i++) {
			const uint8_t *rgba = self->input + 4 * i;
			rgb[i] = rgba[0] / 255.0f;
			rgb[n + i] = rgba[1] / 255.0f;
			rgb[2 * n + i] = rgba[2] / 255.0f;
		}
		pthread_mutex_unlock(&self->lock);
	}];

	NSError *error;
	uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	id<MLFeatureProvider> result = state ? [model predictionFromFeatures:features usingState:state error:&error]
					     : [model predictionFromFeatures:features error:&error];
	MLMultiArray *depth = [result featureValueForName:outputName].multiArrayValue; // 1, H, W
	if (++timed > 10)
		spent += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;
	if (timed == 110) // once per model: whether this Mac keeps up is the first thing to know
		blog(LOG_INFO, "[depth-rgb-trails] the model takes %.1f ms a frame here: the mask follows at up to %.0f frames a second",
		     spent / 1e8, 1e11 / spent);
	NSUInteger rank = depth.shape.count;
	if (rank < 2 || depth.dataType != MLMultiArrayDataTypeFloat16 || depth.shape[rank - 2].intValue != h ||
	    depth.shape[rank - 1].intValue != w) {
		blog(LOG_ERROR, "[depth-rgb-trails] no depth from the model: %s",
		     error ? error.localizedDescription.UTF8String : "its output is not float16 of the input's size");
		[self unload];
		return;
	}
	NSInteger rowStride = depth.strides[rank - 2].integerValue;
	[depth getBytesWithHandler:^(const void *bytes, NSInteger size) {
		pthread_mutex_lock(&self->lock);
		for (int y = 0; y < h; y++)
			memcpy(self->mask + y * w, (const _Float16 *)bytes + y * rowStride, w * sizeof(_Float16));
		self->maskDirty = YES;
		pthread_mutex_unlock(&self->lock);
	}];
}

/// Draws the filter's input into `picture`, at its own size.
- (BOOL)capture:(obs_source_t *)target parent:(obs_source_t *)parent width:(uint32_t)sw height:(uint32_t)sh
{
	if (!picture)
		picture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_texrender_reset(picture);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	BOOL drawn = gs_texrender_begin(picture, sw, sh);
	if (drawn) {
		uint32_t flags = obs_source_get_output_flags(target);
		struct vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)sw, 0.0f, (float)sh, -100.0f, 100.0f);
		if (target == parent && !(flags & (OBS_SOURCE_CUSTOM_DRAW | OBS_SOURCE_ASYNC)))
			obs_source_default_render(target);
		else
			obs_source_video_render(target);
		gs_texrender_end(picture);
	}
	gs_blend_state_pop();
	return drawn;
}

/// The uniforms both techniques read.
- (void)bind:(gs_texture_t *)image
{
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), image);
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "depth"), maskTexture);
	gs_effect_set_float(gs_effect_get_param_by_name(effect, "scale"), focalOverWidth);
	gs_effect_set_float(gs_effect_get_param_by_name(effect, "cutoff"), atomic_load(&cutoff));
}

/// Adds `picture` to the ring, scaled to the model's `w` x `h`, with what is foreground now as its alpha.
- (void)storeWidth:(int)w height:(int)h
{
	if (count && clock - times[head] > 1.0) // the source was out of sight: its old frames are not a trail
		count = 0;
	int next = (head + 1) % RING;
	if (!ring[next])
		ring[next] = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_texrender_reset(ring[next]);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	if (gs_texrender_begin(ring[next], w, h)) {
		gs_texture_t *image = gs_texrender_get_texture(picture);
		// ponytail: the whole source stretched to the model's shape, which is 16:9 in the released packages.
		// A source of another shape gets a distorted frame; export a model of its shape if the depth suffers.
		gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
		[self bind:image];
		while (gs_effect_loop(effect, "Store"))
			gs_draw_sprite(image, 0, w, h);
		gs_texrender_end(ring[next]);
		head = next;
		times[head] = clock;
		if (count < RING)
			count++;
	}
	gs_blend_state_pop();
}

/// Copies the newest frame of the ring into `input`, for the model.
- (BOOL)handOverWidth:(int)w height:(int)h
{
	if (!stage || gs_stagesurface_get_width(stage) != (uint32_t)w || gs_stagesurface_get_height(stage) != (uint32_t)h) {
		gs_stagesurface_destroy(stage);
		stage = gs_stagesurface_create(w, h, GS_RGBA);
	}
	gs_stage_texture(stage, gs_texrender_get_texture(ring[head]));
	uint8_t *data;
	uint32_t linesize;
	if (!gs_stagesurface_map(stage, &data, &linesize))
		return NO;
	pthread_mutex_lock(&lock);
	BOOL fits = width == w && height == h; // not if the model was replaced meanwhile
	for (int y = 0; fits && y < h; y++)
		memcpy(input + (size_t)4 * w * y, data + (size_t)linesize * y, (size_t)4 * w);
	pthread_mutex_unlock(&lock);
	gs_stagesurface_unmap(stage);
	return fits;
}

/// The newest frame of the ring that is at least `age` seconds old, or the oldest it holds.
- (gs_texture_t *)frameAged:(double)age
{
	int index = head;
	for (int back = 0; back < count; back++) {
		index = (head - back + RING) % RING;
		if (clock - times[index] >= age)
			break;
	}
	return gs_texrender_get_texture(ring[index]);
}

- (void)render
{
	obs_source_t *target = obs_filter_get_target(source), *parent = obs_filter_get_parent(source);
	uint32_t sw = target ? obs_source_get_base_width(target) : 0, sh = target ? obs_source_get_base_height(target) : 0;
	pthread_mutex_lock(&lock);
	int w = width, h = height;
	pthread_mutex_unlock(&lock);
	if (!effect || !parent || !sw || !sh || !w) {
		obs_source_skip_video_filter(source);
		return;
	}
	const bool srgb = gs_framebuffer_srgb_enabled();
	gs_enable_framebuffer_srgb(false); // every pass here copies the picture's values as they are

	// Once per OBS frame, however often it is drawn: the picture, then the ring, then the model if it is free.
	// ponytail: the mask is one or two frames behind the picture; delay the picture to match if fast motion shows it.
	if (fresh && [self capture:target parent:parent width:sw height:sh]) {
		fresh = NO;
		pthread_mutex_lock(&lock);
		if (maskDirty && width == w && height == h) {
			if (maskTexture && (gs_texture_get_width(maskTexture) != (uint32_t)w || gs_texture_get_height(maskTexture) != (uint32_t)h)) {
				gs_texture_destroy(maskTexture);
				maskTexture = NULL;
			}
			if (!maskTexture)
				maskTexture = gs_texture_create(w, h, GS_R16F, 1, NULL, GS_DYNAMIC);
			gs_texture_set_image(maskTexture, (const uint8_t *)mask, w * sizeof(_Float16), false);
			maskDirty = NO;
		}
		pthread_mutex_unlock(&lock);
		if (maskTexture) {
			[self storeWidth:w height:h];
			if (count && !atomic_exchange(&busy, true)) {
				if ([self handOverWidth:w height:h])
					dispatch_async(queue, ^{
						[self infer];
						atomic_store(&self->busy, false);
					});
				else
					atomic_store(&busy, false);
			}
		}
	}

	gs_texture_t *image = picture ? gs_texrender_get_texture(picture) : NULL;
	if (!image || !maskTexture || !count) {
		gs_enable_framebuffer_srgb(srgb);
		obs_source_skip_video_filter(source);
		return;
	}
	// From the top: the foreground, its copy one delay ago in red, two in blue, three in green, the background.
	double step = atomic_load(&delay);
	float spread = atomic_load(&blur);
	gs_texture_t *recent = gs_texrender_get_texture(ring[head]);
	struct vec2 texel;
	vec2_set(&texel, spread / gs_texture_get_width(recent), spread / gs_texture_get_height(recent));
	[self bind:image];
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "recent"), recent);
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "ghost1"), [self frameAged:step]);
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "ghost2"), [self frameAged:2 * step]);
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "ghost3"), [self frameAged:3 * step]);
	gs_effect_set_float(gs_effect_get_param_by_name(effect, "opacity"), atomic_load(&opacity));
	gs_effect_set_vec2(gs_effect_get_param_by_name(effect, "texel"), &texel);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
	while (gs_effect_loop(effect, "Draw"))
		gs_draw_sprite(image, 0, sw, sh);
	gs_blend_state_pop();
	gs_enable_framebuffer_srgb(srgb);
}

@end

static const char *filter_name(void *unused)
{
	return "Depth RGB Trails";
}

static void filter_update(void *data, obs_data_t *settings)
{
	TrailsFilter *filter = (__bridge TrailsFilter *)data;
	atomic_store(&filter->cutoff, (float)obs_data_get_double(settings, "cutoff"));
	atomic_store(&filter->delay, (float)obs_data_get_int(settings, "delay") / 1000.0f);
	atomic_store(&filter->opacity, (float)obs_data_get_double(settings, "opacity"));
	atomic_store(&filter->blur, (float)obs_data_get_double(settings, "blur"));
	NSString *path = @(obs_data_get_string(settings, "model")), *function = @(obs_data_get_string(settings, "size"));
	dispatch_async(filter->queue, ^{
		[filter load:path function:function];
	});
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	TrailsFilter *filter = [TrailsFilter new];
	filter->source = source;
	char *path = obs_module_file("rgb_trails.effect");
	char *error = NULL;
	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(path, &error);
	obs_leave_graphics();
	if (!filter->effect)
		blog(LOG_ERROR, "[depth-rgb-trails] cannot load rgb_trails.effect: %s", error ? error : "no such file");
	bfree(path);
	bfree(error);
	filter_update((__bridge void *)filter, settings);
	return (__bridge_retained void *)filter;
}

static void filter_destroy(void *data)
{
	TrailsFilter *filter = (__bridge_transfer TrailsFilter *)data;
	dispatch_sync(filter->queue, ^{ // after the frame in flight
		[filter unload];
	});
	obs_enter_graphics();
	gs_effect_destroy(filter->effect);
	gs_texrender_destroy(filter->picture);
	for (int i = 0; i < RING; i++)
		gs_texrender_destroy(filter->ring[i]);
	gs_stagesurface_destroy(filter->stage);
	gs_texture_destroy(filter->maskTexture);
	obs_leave_graphics();
}

static void filter_tick(void *data, float seconds)
{
	TrailsFilter *filter = (__bridge TrailsFilter *)data;
	filter->clock += seconds;
	filter->fresh = YES;
}

static void filter_render(void *data, gs_effect_t *unused)
{
	[(__bridge TrailsFilter *)data render];
}

static void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_double(settings, "cutoff", 1.0);
	obs_data_set_default_int(settings, "delay", 150);
	obs_data_set_default_double(settings, "opacity", 0.6);
	obs_data_set_default_double(settings, "blur", 0.0);
}

static obs_properties_t *filter_properties(void *unused)
{
	obs_properties_t *properties = obs_properties_create();
	obs_properties_add_path(properties, "model", "Model (.mlpackage)", OBS_PATH_FILE,
				"Core ML model (*.mlpackage *.mlmodelc *.json)", NULL);
	obs_property_t *size = obs_properties_add_list(properties, "size", "Model input size", OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(size, "The model's default", "");
	// The sizes of the packages Apple publishes; a package exported at one size only has its default.
	for (NSString *name in @[@"640x360", @"854x480", @"896x512", @"1138x640", @"1280x720"])
		obs_property_list_add_string(size, name.UTF8String, [@"res_" stringByAppendingString:name].UTF8String);
	obs_properties_add_float_slider(properties, "cutoff", "Foreground ends at (approximate meters)", 0.2, 10.0, 0.05);
	obs_properties_add_int_slider(properties, "delay", "Delay between layers (ms)", 20, 500, 10);
	obs_properties_add_float_slider(properties, "opacity", "Layer opacity", 0.1, 1.0, 0.05);
	obs_properties_add_float_slider(properties, "blur", "Background blur (0 is off)", 0.0, 4.0, 0.25);
	return properties;
}

bool obs_module_load(void)
{
	static struct obs_source_info info = {
		.id = "depth_rgb_trails",
		.type = OBS_SOURCE_TYPE_FILTER,
		.output_flags = OBS_SOURCE_VIDEO,
		.get_name = filter_name,
		.create = filter_create,
		.destroy = filter_destroy,
		.update = filter_update,
		.get_defaults = filter_defaults,
		.get_properties = filter_properties,
		.video_tick = filter_tick,
		.video_render = filter_render,
	};
	obs_register_source(&info);
	return true;
}
