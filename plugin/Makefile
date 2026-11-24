# Builds the OBS plugin for Apple silicon: `make`, `make install`, `make zip`.
#
# The plugin links against the libobs inside OBS.app, and compiles against the headers of the same OBS version,
# fetched here with the one header-only library they need on Apple silicon.

OBS_VERSION ?= 33.0.3
VERSION ?= 0.0.0
OBS_APP ?= /Applications/OBS.app
NAME := depth-rgb-trails
BUNDLE := build/$(NAME).plugin
OBS_SRC := build/obs-studio
SIMDE := build/simde
PLUGINS := $(HOME)/Library/Application Support/obs-studio/plugins

$(BUNDLE): plugin.m rgb_trails.effect Info.plist | $(OBS_SRC) $(SIMDE) build/include/obsconfig.h
	rm -rf $(BUNDLE)
	mkdir -p $(BUNDLE)/Contents/MacOS $(BUNDLE)/Contents/Resources
	clang -bundle -fobjc-arc -O2 -Wall -arch arm64 -mmacosx-version-min=15.0 \
		-I$(OBS_SRC)/libobs -Ibuild/include -I$(SIMDE) \
		-F"$(OBS_APP)/Contents/Frameworks" -framework libobs -framework CoreML -framework Foundation \
		plugin.m -o $(BUNDLE)/Contents/MacOS/$(NAME)
	cp Info.plist $(BUNDLE)/Contents/
	plutil -replace CFBundleShortVersionString -string $(VERSION) $(BUNDLE)/Contents/Info.plist
	plutil -replace CFBundleVersion -string $(VERSION) $(BUNDLE)/Contents/Info.plist
	cp rgb_trails.effect $(BUNDLE)/Contents/Resources/
	codesign --force --sign - $(BUNDLE)

$(OBS_SRC):
	git clone -q -c advice.detachedHead=false --depth 1 --branch $(OBS_VERSION) --filter=blob:none --sparse https://github.com/obsproject/obs-studio $@
	git -C $@ sparse-checkout set libobs

$(SIMDE):
	git clone -q -c advice.detachedHead=false --depth 1 --branch v0.8.2 https://github.com/simd-everywhere/simde $@

# What OBS's own build writes from libobs/obsconfig.h.in.
build/include/obsconfig.h:
	mkdir -p build/include
	printf '#pragma once\n#define OBS_RELEASE_CANDIDATE 0\n#define OBS_BETA 0\n' > $@

install: $(BUNDLE)
	mkdir -p "$(PLUGINS)"
	rm -rf "$(PLUGINS)/$(NAME).plugin"
	cp -R $(BUNDLE) "$(PLUGINS)/"

zip: $(BUNDLE)
	cd build && rm -f $(NAME)-macos-arm64.zip && zip -qry $(NAME)-macos-arm64.zip $(NAME).plugin

clean:
	rm -rf build

.PHONY: install zip clean
