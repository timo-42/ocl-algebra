extern crate ocl_algebra;

use ocl_algebra::*;

fn main() {
    let eps = 1.0e-6;
    let mut c = new().unwrap();
    let m0 = Matrix{rows: 10000, cols: 10000, data: vec![3.0; 10000*10000]};
    let m1 = Matrix{rows: 10000, cols: 1, data: vec![4.0; 10000]};
    
    /* matrix multiplication
       [1 2] * [4] = [14]
       [3 4]   [5]   [32] */
    let m = c.mul_matrix_matrix(&m0,&m1);
    println!("{:?}", m.data);
}
