NAME = test
N_CORES = 32
GL_PATH = ./target/dev-optim/gammaloop
AMPLITUDE = ./gl_states/$(NAME)/processes/amplitudes/$(NAME)
OUTPUT_STATE = ./gl_states/$(NAME)_output
INTEGRATION_STATE = ./gl_states/integration_$(NAME)
INTEGRATION_RESULTS = $(INTEGRATION_STATE)/integration_results.txt
RUNCARD = ./runcards/generate_$(NAME).toml

build_all: build_gammaloop build_gammaloop_python_api

build_gammaloop:
	cargo build -p gammaloop-api --bin gammaloop --features ufo_support --profile dev-optim

build_gammaloop_python_api:
	maturin develop -m gammaloop-api/Cargo.toml --features=ufo_support,python_api --release

generate:
	echo "$(NAME)"
	rm -rf $(OUTPUT_STATE); $(GL_PATH) -o -s $(OUTPUT_STATE) -t generate $(RUNCARD)

integrate:
	echo "$(NAME)"
	rm -rf $(INTEGRATION_STATE); GL_DISPLAY_FILTER=info GL_LOGFILE_FILTER=warning $(GL_PATH) -n -s $(OUTPUT_STATE) integrate --workspace-path $(INTEGRATION_STATE) --result-path $(INTEGRATION_RESULTS) -c $(N_CORES)
