use rand::{Rng, thread_rng};
use polars::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use std::fs::File;
use std::sync::Arc;


pub fn mnist_df() -> Result<DataFrame> {
    let file = File::open("mnist.csv")
                    .expect("could not read file");

    CsvReader::new(file)
            .infer_schema(None)
            .has_header(true)
            .finish()
}

pub fn init_params() -> (
        ndarray_rand::rand_distr::Normal<f64>, 
        ndarray_rand::rand_distr::Normal<f64>, 
        ndarray_rand::rand_distr::Normal<f64>, 
        ndarray_rand::rand_distr::Normal<f64>
    ){
    let mut rng = thread_rng();
    let mut params = Array2::<f64>::zeros((784, 10));
    let w1 = Normal::new(10.0, 784.0).unwrap();
    let b1 = Normal::new(10.0, 1.0).unwrap();
    let w2 = Normal::new(10.0, 10.0).unwrap();
    let b2 = Normal::new(10.0, 1.0).unwrap();

    (w1, b1, w2, b2)
}

// pub fn init_params() -> ndarray::Array2<f64> {
//     let mut rng = thread_rng();
//     let mut params = Array2::<f64>::zeros((784, 10));
//     for i in 0..10 {
//         let mut row = Array1::<f64>::zeros(784);
//         for j in 0..784 {
//             row[j] = rng.gen_range(-0.1, 0.1);
//         }
//         params.slice_mut(s![.., i]).assign(&row);
//     }
//     params
// }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);

        let df = mnist_df().unwrap();
        println!("{:?}", df.head(Some(10)));  
        let data = df.to_ndarray::<UInt64Type>().unwrap();

        //covert to u8
        let data = data.mapv(|x| x as u8);
        let m: usize = data.shape()[0] as usize;
        let n: usize = data.shape()[1] as usize;

        let total = 1000;

        println!("{:?}", (m, n));

        let data_dev = data.slice(s![0..total, 1]).t();
        let ydev = data.slice(s![0, 1]);
        print!("{:?}", ydev);

        
        let xdev = data.slice(s![1..n, 1]);
        print!("{:?}", xdev);

        let data_train = data.slice(s![total..m, 1]);
        let data_train = data_train.t();
        
        let y_train = data_train.slice(s![0]);
        print!("{:?}", y_train);
        let x_train = data_train.slice(s![1..n]);
        print!("{:?}", x_train);


    }
}
