# ocl-algebra
Low Level Linear Algebra Library for OpenCL

## goals
 * simple low level linear algebra api
 * implement an synchron api (copy data to device, math operation, copy result to host and delete data on device)
 * todo: implement asynchron api, run multiple math operation without always copying data to/from the device
 * library may be useful as alternative backend for pure rust alebra libraries

## example
From [`examples/simple.rs`]:
```rust
extern crate ocl_algebra;

use ocl_algebra::*;

func main() {
    // init library
    let mut c = new();
    
    // init some matrices
    let m0 = Matrix{rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0]};
    let m1 = Matrix{rows: 2, cols: 1, data: vec![4.0, 5.0]};
    
    /* matrix multiplication
       [1 2] * [4] = [14]
       [3 4]   [5]   [32] */
    let m = c.mul_matrix_matrix(&m0,&m1);
    
    /* matrix scalar multiplication 
       [1 2] * 1.5 = [1.5 3]
       [3 4]         [4.5 6]*/
    let m = c.mul_matrix_scalar(&m0,1.5);
}
```

## Development

#### License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

#### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

<br/>

*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.”* *“Vulkan and the Vulkan logo are trademarks of the Khronos Group Inc.”*
