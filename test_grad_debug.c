#include <stdio.h>
#include <stdlib.h>

int main() {
    // Test case: x.sum() gradient w.r.t. unrelated w
    // x = [1, 2, 3], w = random tensor
    
    // Create x
    float x_data[] = {1.0f, 2.0f, 3.0f};
    
    // Create unrelated w  
    float w_data[] = {0.5f, -0.2f, 0.8f};
    
    // x.sum() = 6.0
    float x_sum = 6.0f;
    
    // Gradient of x.sum() w.r.t. x is [1, 1, 1]
    // Gradient of x.sum() w.r.t. w should be undefined (not in graph)
    
    printf("x.sum() = %.1f\n", x_sum);
    printf("Gradient w.r.t. x should be [1, 1, 1]\n");
    printf("Gradient w.r.t. w should be NULL (not in graph)\n");
    
    // The issue is that our code might be returning a zero gradient
    // instead of NULL for unrelated tensors
    
    return 0;
}
