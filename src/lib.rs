extern crate ocl;

mod kernel;

use ocl::{Platform, Device, ProQue, Buffer};
use ocl::core;
use ocl::core::{DeviceInfo};

pub struct Context{
    pub compute_units: u32,
    pro_que:       ocl::ProQue,
}


pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Context {
    pub fn mul_matrix_scalar(&mut self, matrix: &Matrix, scalar: f32) -> Matrix {
        let ref mut ocl_pq = self.pro_que;
        
        // set work dimension, we can work with 1
        ocl_pq.set_dims([matrix.rows*matrix.cols]);
        
        // copy matrix to device
        let source_buffer = Buffer::new(
            &ocl_pq.queue().clone(),
            Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR), 
            ocl_pq.dims().clone(), 
            Some(&matrix.data)).unwrap();
        
        // prepare result matrix
        let mut result = vec![0.0f32; matrix.cols*matrix.rows];
        let result_buffer: Buffer<f32> = ocl_pq.create_buffer().unwrap();

        // Create a kernel with arguments corresponding to those in the kernel:
        let kernel = ocl_pq.create_kernel("mul_matrix_scalar").unwrap()
            .arg_scl(scalar)
            .arg_buf(&source_buffer)
            .arg_buf(&result_buffer);

        // Enqueue kernel:
        kernel.enq().unwrap(); // send kernel to device and run it

        // Read results from the device into result_buffer's local vector:
        result_buffer.read(&mut result).enq().unwrap();
        
        // return matrix
        Matrix{rows: matrix.rows,cols: matrix.cols, data: result}
    }
    
    pub fn mul_matrix_matrix(&mut self, matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
        let ref mut ocl_pq = self.pro_que;
        
        // set work dimension for matrix_a
        ocl_pq.set_dims([matrix_a.rows,matrix_a.cols]);
        
        // copy matrix_a to device
        let matrix_a_buffer = Buffer::new(
            &ocl_pq.queue().clone(),
            Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR), 
            ocl_pq.dims().clone(), 
            Some(&matrix_a.data)).unwrap();

        // set work dimension for matrix_b
        ocl_pq.set_dims([matrix_b.rows,matrix_b.cols]);
        // copy matrix_b to device
        let matrix_b_buffer = Buffer::new(
            &ocl_pq.queue().clone(),
            Some(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR), 
            ocl_pq.dims().clone(), 
            Some(&matrix_b.data)).unwrap();
        
        // set work dimension for result matrix
        ocl_pq.set_dims([matrix_a.rows,matrix_b.cols]);
        // prepare result matrix
        
        let mut result = vec![0.0f32; matrix_a.rows*matrix_b.cols];
        let result_buffer: Buffer<f32> = ocl_pq.create_buffer().unwrap();

        // Create a kernel with arguments corresponding to those in the kernel:
        let kernel = ocl_pq.create_kernel("mul_matrix_matrix").unwrap()
            .arg_buf(&matrix_a_buffer)
            .arg_buf(&matrix_b_buffer)
            .arg_buf(&result_buffer)
            .arg_scl(matrix_a.rows as i32)
            .arg_scl(matrix_a.cols as i32)
            .arg_scl(matrix_b.cols as i32);
        //panic!("{}",kernel.get_lws().to_len());

        // Enqueue kernel:
        kernel.enq().unwrap(); // send kernel to device and run it

        // Read results from the device into result_buffer's local vector:
        result_buffer.read(&mut result).enq().unwrap();
        
        // return matrix
        Matrix{rows: matrix_a.rows, cols: matrix_b.cols, data: result}
    }
}

pub fn new() -> Option<Context> {
    // wich Device should we choose?
    // the one with the most Compute units!
    let mut compute_units = 0;
    let mut ocl_device = None;
    let platforms = Platform::list();
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        let devices = Device::list_all(platform);
        for d_idx in 0..devices.len() {
            let device = devices[d_idx];
            let deviceinforesult = core::get_device_info(&device, DeviceInfo::MaxComputeUnits);
            let units = deviceinforesult.to_string().parse().unwrap();
            if units > compute_units {
                ocl_device = Some(device);
                compute_units = units;
            }
        }
    }
    // something went wrong no, opencl not installed
    if compute_units == 0 {
        return None
    }
    let que = ProQue::builder()
              .device(ocl_device.unwrap())
              .src(kernel::OCL_KERNEL)
              .build().expect("Build ProQue");
    Some(Context{
        compute_units: compute_units,
        pro_que : que,
    })
}

#[test]
fn single_test() {
    let eps = 1.0e-6;
    let mut c = new().unwrap();
    let m0 = Matrix{rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0]};
    let m1 = Matrix{rows: 2, cols: 1, data: vec![4.0, 5.0]};
    
    /* matrix multiplication
       [1 2] * [4] = [14]
       [3 4]   [5]   [32] */
    let m = c.mul_matrix_matrix(&m0,&m1);
    assert!((m.data[0] - 14.0f32).abs() < eps);
    assert!((m.data[1] - 32.0f32).abs() < eps);
    
    /* matrix scalar multiplication 
       [1 2] * 1.5 = [1.5 3]
       [3 4]         [4.5 6]*/
    let m = c.mul_matrix_scalar(&m0,1.5);
    assert!( (m.data[0] - 1.5f32) < eps);
    assert!( (m.data[1] - 3.0f32) < eps);
    assert!( (m.data[2] - 4.5f32) < eps);
    assert!( (m.data[3] - 6.0f32) < eps);
}
