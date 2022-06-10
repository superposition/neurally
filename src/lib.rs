use rand::{thread_rng};
use polars::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Normal, NormalError, Distribution};
use ndarray_rand::rand::{Rng, RngCore};
use ndarray_rand::RandomExt;
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
//test init_params 

#[cfg(tests)]
mod tests {
    use super::*;

    #[test]
    fn test_init_params() {
        let intial_params = init_params();
        assert_eq!(intial_params.len(), 4);
    }
}


pub fn init_params() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let w1 = Array::random((10, 784), normal);
    let b1 = Array::random(10, normal);
    let w2 = Array::random((10, 10), normal);
    let b2 = Array::random(10, normal);
    
    (w1, b1, w2, b2)
}

pub fn ReLU(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let mut x = x.clone();
    let mut max = x[0];
    for i in 1..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }
    x = x - max;
    x = x.mapv(|x| ReLU(x));
    let sum = x.iter().sum::<f64>();
    x = x / sum;
    x
}


pub fn forward_prop(x: Array1<f64>, params: (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>){
    let (w1, b1, w2, b2) = params;
    let mut z1 = w1.dot(&x) + &b1;
    let mut a1 = z1.mapv(|x| ReLU(x));
    let mut z2 = w2.dot(&a1) + &b2;
    let mut a2 = softmax(&z2);

    (z1, a1, z2, a2)
}

pub fn one_hot(y: Array1<u8>) -> Array2<u8> {
    let y = y.clone();
    let mut y_one_hot = Array::zeros((y.shape()[0], 10));
    for i in 0..y.len() {
        y_one_hot[(i, y[i] as usize)] = 1;
    }
    y_one_hot.t().to_owned()

}


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
