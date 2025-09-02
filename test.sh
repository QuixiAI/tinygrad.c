#!/bin/bash

# Build if needed
[ ! -f "build/test_tensor" ] && ./build.sh

# Initialize counters
total_tests=0
total_passed=0
total_failed=0
total_ignored=0
failed_details=""
crashed_suites=""

# Count total test files for header
test_count=$(ls build/test_* 2>/dev/null | wc -l)

# Run all tests in parallel if requested
if [ "$1" = "--parallel" ] || [ "$1" = "-p" ]; then
    echo "Running tests in parallel..."
    for test in build/test_*; do
        [ -x "$test" ] && "$test" &
    done
    wait
else
    # Run tests sequentially with dot-reporter style
    echo "Running $test_count test suites..."
    
    # Run tests and show dots/F for each test function
    for test in build/test_*; do
        if [ -x "$test" ]; then
            output=$("$test" 2>&1)
            exit_code=$?
            
            # Check for segfault or core dump
            if echo "$output" | grep -q "Segmentation fault\|core dumped" || [ "$exit_code" -eq 139 ] || [ "$exit_code" -eq 134 ]; then
                suite_name=$(basename "$test")
                crashed_suites="$crashed_suites  $suite_name (CRASHED)\n"
                printf "X"
                continue
            fi
            
            # Parse Unity output to show dots for each test
            has_test_output=false
            if echo "$output" | grep -q ":PASS\|:FAIL"; then
                has_test_output=true
                # Process each test result line
                while IFS= read -r line; do
                    if echo "$line" | grep -q ":PASS$"; then
                        printf "."
                        total_passed=$((total_passed + 1))
                        total_tests=$((total_tests + 1))
                    elif echo "$line" | grep -q ":FAIL"; then
                        printf "F"
                        total_failed=$((total_failed + 1))
                        total_tests=$((total_tests + 1))
                        # Extract test name and failure details
                        test_name=$(echo "$line" | sed 's/:FAIL.*//')
                        test_name=$(basename "$test_name")
                        suite_name=$(basename "$test")
                        # Get the failure message if available
                        fail_msg=$(echo "$line" | sed 's/.*:FAIL[: ]*//')
                        if [ -z "$fail_msg" ] || [ "$fail_msg" = "$line" ]; then
                            failed_details="$failed_details\n  ✗ $suite_name :: $test_name"
                        else
                            failed_details="$failed_details\n  ✗ $suite_name :: $test_name\n      $fail_msg"
                        fi
                    fi
                done <<< "$output"
            fi
            
            # If no test output found but didn't crash, mark as error
            if [ "$has_test_output" = "false" ] && [ "$exit_code" -ne 0 ]; then
                suite_name=$(basename "$test")
                crashed_suites="$crashed_suites  $suite_name (ERROR - no test output)\n"
                printf "E"
            fi
            
            # Parse Unity summary for ignored tests
            if echo "$output" | grep -q "Tests.*Failures.*Ignored"; then
                summary=$(echo "$output" | grep "Tests.*Failures.*Ignored" | tail -1)
                ignored=$(echo "$summary" | sed -n 's/.*Failures \([0-9]*\) Ignored.*/\1/p')
                if [ -n "$ignored" ] && [ "$ignored" -gt 0 ]; then
                    total_ignored=$((total_ignored + ignored))
                    # Show 'i' for each ignored test
                    for ((i=1; i<=ignored; i++)); do
                        printf "i"
                    done
                fi
            fi
        fi
    done
    
    # Print summary
    echo ""
    echo ""
    if [ "$total_failed" -eq 0 ] && [ -z "$crashed_suites" ]; then
        echo "✓ $total_passed tests passed"
    else
        echo ""
        echo "Tests: $total_failed failed, $total_passed passed, $total_ignored ignored, $total_tests total"
        
        if [ -n "$failed_details" ] || [ -n "$crashed_suites" ]; then
            echo ""
            echo "Failed tests:"
            if [ -n "$failed_details" ]; then
                echo -e "$failed_details"
            fi
            if [ -n "$crashed_suites" ]; then
                echo ""
                echo "Crashed suites:"
                echo -e "$crashed_suites"
            fi
        fi
    fi
    
    # Exit with error if any tests failed
    if [ "$total_failed" -gt 0 ] || [ -n "$crashed_suites" ]; then
        exit 1
    fi
fi