all: html

.SECONDEXPANSION:
build/%.ipynb: %.md $$(wildcard %*.json)
	@echo $^
	@mkdir -p $(@D)
	@cp $^ $(@D)
	cd $(@D); rm ../../build/$<; python ../md2ipynb.py ../../$< ../../$@

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

MARKDOWN = $(wildcard */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard */*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

# ORIGN_DEPS = $(wildcard img/* data/*) environment.yml utils.py README.md
# DEPS = $(patsubst %, build/%, $(ORIGN_DEPS))


html: $(OBJ)
	@echo $(OBJ)
	make -C build html


# clean:
	# rm -rf build/* $(DEPS) $(PKG)
