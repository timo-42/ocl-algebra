
pub static OCL_KERNEL: &'static str = r#"
__kernel void mul_matrix_scalar(
            __private float const coeff,
            __global float const* const src,
            __global float* const res)
{
    uint const idx = get_global_id(0);
    res[idx] = src[idx] * coeff;
}

__kernel void mul_matrix_matrix(
        const __global float* A, 
        const __global float* B,
              __global float* C, 
        const int A_rows,
        const int A_cols,
        const int B_cols)
{
  
   int C_row = get_global_id(0); 
   int C_col = get_global_id(1);
   //    C[i,     k    ] = SUM A[i    ,k] * B[k,j    ]
   // => C[C_row, C_col] = SUM A[C_row,k] * B[k,C_col]
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < A_cols; k++)
   {
      float elementA = A[C_row * A_cols + k];
      float elementB = B[k     * B_cols + C_col];
      value += elementA * elementB;
   }
 

   C[C_col * A_rows + C_row] = value;
   
}
"#;
