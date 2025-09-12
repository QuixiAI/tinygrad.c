SHELL := /bin/bash
DOCKER_IMAGE ?= conanio/gcc11-ubuntu16.04:2.20.1
PYTHON ?= python3
VENV_DIR ?= .venv

# Quiet by default. Set QUIET=0 for verbose logs.
QUIET ?= 1
ASAN ?= 0
ifeq ($(QUIET),1)
  MAKEFLAGS += -s
  CMAKE_BUILD_SILENT := -- -s
  CONAN_LOG_LEVEL := -v error
  REDIR := >
else
  CMAKE_BUILD_SILENT :=
  CONAN_LOG_LEVEL :=
  REDIR :=
endif

# Conan configuration
# If using local Conan (default), we prefer a project-local cache unless PERSIST_CONAN=1.
# When PERSIST_CONAN=1, use the user's home cache (~/.conan2). CLEAN_CONAN=1 clears that cache first.
CONAN_CACHE_DIR ?= $(PWD)/build/.conan2
# Use project-local Conan cache by default; users can opt-in to home cache with PERSIST_CONAN=1
CONAN_HOME ?= $(if $(filter 1,$(PERSIST_CONAN)),$(HOME)/.conan2,$(CONAN_CACHE_DIR))
CONAN_BIN ?= $(if $(wildcard $(PWD)/$(VENV_DIR)/bin/conan),$(PWD)/$(VENV_DIR)/bin/conan,conan)
TOOLCHAIN_PATH := build/conan/build/Release/generators/conan_toolchain.cmake

.PHONY: all build rebuild test clean realclean setup conan

all: build

# Build (keeps build/ by default). Use CLEAN=1 to force a fresh build dir.
build:
	@set -euo pipefail; \
	if [ "${CLEAN:-0}" = "1" ]; then rm -rf build; fi; \
	mkdir -p build/logs; \
	$(MAKE) $(TOOLCHAIN_PATH) || { echo "Conan failed. See build/logs/conan_install.log"; exit 1; }; \
	echo "[cmake] configuring..."; \
	cmake -S . -B build \
	  -DBUILD_TESTS=ON \
	  -DBUILD_EXAMPLES=ON \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_PATH) \
	  -DCMAKE_C_FLAGS="-DUNITY_INCLUDE_DOUBLE -DUNITY_INCLUDE_FLOAT" \
	  $(if $(filter 1,$(ASAN)),-DTG_ENABLE_ASAN=ON,-DTG_ENABLE_ASAN=OFF) \
	  > build/logs/cmake_configure.log 2>&1 || { echo "CMake configure failed. See build/logs/cmake_configure.log"; exit 1; }; \
	echo "[cmake] building..."; \
	cmake --build build -j $(CMAKE_BUILD_SILENT)

rebuild: 
	$(MAKE) CLEAN=1 build

# Bootstrap: create a Python venv and install Conan locally, then resolve deps
setup: $(VENV_DIR)/bin/conan
	@$(MAKE) conan

$(VENV_DIR)/bin/conan:
	@set -euo pipefail; \
	if [ ! -d "$(VENV_DIR)" ]; then \
	  echo "[setup] creating venv at $(VENV_DIR)"; \
	  $(PYTHON) -m venv "$(VENV_DIR)"; \
	fi; \
	"$(VENV_DIR)/bin/pip" install -U pip conan

# Explicit Conan install to produce the CMake toolchain and deps
conan: $(TOOLCHAIN_PATH)

$(TOOLCHAIN_PATH): conanfile.txt profiles/linux-gcc11 profiles/locks/linux-gcc11.lock
	@set -euo pipefail; \
	# Determine which 'conan' to use: prefer project venv, else system
	if [ -x "$(CONAN_BIN)" ]; then CONAN_CMD="$(CONAN_BIN)"; \
	elif command -v conan >/dev/null 2>&1; then CONAN_CMD="conan"; \
	else \
	  echo "conan not found. Please install Conan 2 into your Python env (e.g., 'python3 -m venv .venv && . .venv/bin/activate && pip install conan') or install system-wide."; \
	  exit 1; \
	fi; \
	LOCK_DIR=build/conan/locks; \
	LOCK_OUT=$$LOCK_DIR/linux-gcc11.lock; \
	LOCK_SRC=profiles/locks/linux-gcc11.lock; \
	mkdir -p "$$LOCK_DIR" build/logs; \
	echo "[conan] using: $$CONAN_CMD (CONAN_HOME=$(CONAN_HOME))"; \
	( \
	  export CONAN_HOME="$(CONAN_HOME)"; \
	  if [ "${CLEAN_CONAN:-0}" = "1" ]; then rm -rf "$$CONAN_HOME"/*; fi; \
	  mkdir -p "$$CONAN_HOME/profiles"; \
	  if [ ! -f "$$CONAN_HOME/profiles/default" ]; then "$$CONAN_CMD" profile detect --force $(CONAN_LOG_LEVEL); fi; \
	  if ! "$$CONAN_CMD" lock create . --profile=profiles/linux-gcc11 --lockfile-out="$$LOCK_OUT" $(CONAN_LOG_LEVEL); then \
	    echo "[conan] lock create failed; will try existing lockfile and cached packages"; \
	  fi; \
	  if [ -f "$$LOCK_OUT" ]; then USE_LOCK="--lockfile=$$LOCK_OUT"; \
	  elif [ -f "$$LOCK_SRC" ]; then USE_LOCK="--lockfile=$$LOCK_SRC"; \
	  else USE_LOCK=""; fi; \
  if ! "$$CONAN_CMD" install . \
	      -of build/conan \
	      $$USE_LOCK \
	      --profile=profiles/linux-gcc11 \
	      --build=unity/* \
	      --build=missing \
	      -g CMakeDeps -g CMakeToolchain \
	      --deployer=full_deploy \
	      --deployer-folder build/conan/full_deploy \
	      $(CONAN_LOG_LEVEL); then \
	    echo "[conan] online install failed; retrying with --no-remote using cached packages"; \
    "$$CONAN_CMD" install . \
      -of build/conan \
      $$USE_LOCK \
      --profile=profiles/linux-gcc11 \
      --build=never \
      --no-remote \
      -g CMakeDeps -g CMakeToolchain \
      --deployer=full_deploy \
      --deployer-folder build/conan/full_deploy \
      $(CONAN_LOG_LEVEL); \
	  fi; \
	) > build/logs/conan_install.log 2>&1

test:
	@set -e; \
	if [ ! -x build/test_tensor ]; then \
	  echo "Building tests quietly..."; \
	  ($(MAKE) -s build > /dev/null 2>&1) || { echo "Build failed. Run 'make build' for full logs."; exit 1; }; \
	fi; \
	bash -lc ' \
	  total_tests=0; total_passed=0; total_failed=0; total_ignored=0; failed_details=""; crashed_suites=""; \
	  test_count=$$(ls build/test_* 2>/dev/null | wc -l); \
	  echo "Running $$test_count test suites..."; \
	  for test in build/test_*; do \
	    if [ -x "$$test" ]; then \
	      output=$$("$$test" 2>&1); exit_code=$$?; \
	      if [ "$$exit_code" -gt 128 ]; then \
	        suite_path=$$(echo "$$test" | sed "s|^build/test_||"); \
	        if [ -f "tests/uop/test_$${suite_path}.c" ]; then suite_file="tests/uop/test_$${suite_path}.c"; \
	        elif [ -f "tests/test_$${suite_path}.c" ]; then suite_file="tests/test_$${suite_path}.c"; \
	        else suite_file="tests/test_$${suite_path}.c"; fi; \
	        signal_num=$$((exit_code - 128)); \
	        case $$signal_num in \
	          1) signal_name="SIGHUP" ;; 2) signal_name="SIGINT" ;; 3) signal_name="SIGQUIT" ;; \
	          4) signal_name="SIGILL" ;; 5) signal_name="SIGTRAP" ;; 6) signal_name="SIGABRT" ;; \
	          7) signal_name="SIGBUS" ;; 8) signal_name="SIGFPE" ;; 9) signal_name="SIGKILL" ;; \
	          10) signal_name="SIGUSR1" ;; 11) signal_name="SIGSEGV" ;; 12) signal_name="SIGUSR2" ;; \
	          13) signal_name="SIGPIPE" ;; 14) signal_name="SIGALRM" ;; 15) signal_name="SIGTERM" ;; \
	          *) signal_name="signal $$signal_num" ;; \
	        esac; \
	        crashed_suites="$$crashed_suites  $$suite_file ($$signal_name)\n"; \
	        printf "X"; \
	        continue; \
	      fi; \
	      has_test_output=false; \
	      if echo "$$output" | grep -q ":PASS\|:FAIL"; then \
	        has_test_output=true; \
	        while IFS= read -r line; do \
	          if echo "$$line" | grep -q ":PASS$$"; then \
	            printf "."; total_passed=$$((total_passed + 1)); total_tests=$$((total_tests + 1)); \
	          elif echo "$$line" | grep -q ":FAIL"; then \
	            printf "F"; total_failed=$$((total_failed + 1)); total_tests=$$((total_tests + 1)); \
	            full_path=$$(echo "$$line" | cut -d: -f1); \
	            line_num=$$(echo "$$line" | cut -d: -f2); \
	            test_name=$$(echo "$$line" | cut -d: -f3); \
	            suite_name=$$(basename "$$test"); \
	            rel_path=$$(echo "$$full_path" | sed "s|$$PWD/||"); \
	            fail_msg=$$(echo "$$line" | sed "s/.*:FAIL[: ]*//"); \
	            if [ -z "$$fail_msg" ] || [ "$$fail_msg" = "$$line" ]; then \
	              failed_details="$$failed_details\n  ✗ $$suite_name :: $$rel_path:$$line_num:$$test_name"; \
	            else \
	              failed_details="$$failed_details\n  ✗ $$suite_name :: $$rel_path:$$line_num:$$test_name\n      $$fail_msg"; \
	            fi; \
	          fi; \
	        done <<< "$$output"; \
	      fi; \
	      if [ "$$has_test_output" = "false" ] && [ "$$exit_code" -ne 0 ]; then \
	        suite_name=$$(basename "$$test"); \
	        crashed_suites="$$crashed_suites  $$suite_name (ERROR - no test output)\n"; \
	        printf "E"; \
	      fi; \
	      if echo "$$output" | grep -q "Tests.*Failures.*Ignored"; then \
	        summary=$$(echo "$$output" | grep "Tests.*Failures.*Ignored" | tail -1); \
	        ignored=$$(echo "$$summary" | sed -n "s/.*Failures \\([0-9]*\\) Ignored.*/\\1/p"); \
	        if [ -n "$$ignored" ] && [ "$$ignored" -gt 0 ]; then \
	          total_ignored=$$((total_ignored + ignored)); \
	          for ((i=1; i<=ignored; i++)); do printf "i"; done; \
	        fi; \
	      fi; \
	    fi; \
	  done; \
	  echo ""; echo ""; \
	  if [ "$$total_failed" -eq 0 ] && [ -z "$$crashed_suites" ]; then \
	    echo "✓ $$total_passed tests passed"; \
	  else \
	    echo ""; \
	    echo "Tests: $$total_failed failed, $$total_passed passed, $$total_ignored ignored, $$total_tests total"; \
	    if [ -n "$$failed_details" ] || [ -n "$$crashed_suites" ]; then \
	      echo ""; echo "Failed tests:"; \
	      if [ -n "$$failed_details" ]; then echo -e "$$failed_details"; fi; \
	      if [ -n "$$crashed_suites" ]; then echo ""; echo "Crashed suites:"; echo -e "$$crashed_suites"; fi; \
	    fi; \
	    exit 1; \
	  fi'

.PHONY: ctest
ctest:
	@set -e; \
	if [ ! -d build ]; then $(MAKE) -s build > /dev/null; fi; \
	ctest --test-dir build --output-on-failure

clean:
	rm -rf build

realclean: clean
