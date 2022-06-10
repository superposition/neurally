use rand::{thread_rng};
use polars::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Normal, NormalError, Distribution};
use ndarray_rand::rand::{Rng, RngCore};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
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


pub fn forward_prop(
    w1: Array2<f64>, b1: Array1<f64>, w2: Array2<f64>, b2: Array1<f64>, x: Array1<f64>
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let z1 = w1.dot(&x) + &b1;
    let a1 = z1.mapv(|x| ReLU(x));
    let z2 = w2.dot(&a1) + &b2;
    let a2 = softmax(&z2);

    (z1, a1, z2, a2)
}

pub fn one_hot(y: Array1<f64>) -> Array2<f64> {
    let y = y.clone();
    let mut y_one_hot = Array::zeros((y.shape()[0], 10));
    for i in 0..y.len() {
        y_one_hot[(i, y[i] as usize)] = 1.0;
    }
    y_one_hot.t().to_owned()

}

pub fn ReLU_derivative(x: Array1<f64>) -> Array1<f64> {
    let mut x = x.clone();
    for i in 0..x.len() {
        if x[i] < 0.0 {
            x[i] = 0.0;
        }
    }
    x
}

// backprop 
pub fn backward_prop(
    z1: Array1<f64>, 
    a1: Array1<f64>, 
    a2: Array1<f64>,  
    w2: Array2<f64>,
    x: Array1<f64>, 
    y: Array1<f64>
) -> (Array1<f64>, f64, Array1<f64>, f64) {
    let m = a2.shape()[0] as f64;
    let one_hot_y = one_hot(y);
    let dz2 = a2 - &one_hot_y;
    let dw2 = 1.0 / m * dz2.dot(&a1.t());
    let db2 = 1.0 / m * dz2.sum();
    let dz1 = w2.t().dot(&dz2) * &ReLU_derivative(z1);
    let dw1 = 1.0 / m * dz1.dot(&x.t());
    let db1 = 1.0 / m * dz1.sum();
    (dw1, db1, dw2, db2)
}

pub fn update_params(
    w1: Array2<f64>, 
    b1: Array1<f64>, 
    w2: Array2<f64>, 
    b2: Array1<f64>, 
    dw1: Array1<f64>, 
    db1: f64, 
    dw2: Array1<f64> , 
    db2:f64, 
    alpha: f64
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let w1 = w1 - alpha * dw1;
    let b1 = b1 - alpha * db1;
    let w2 = w2 - alpha * dw2;
    let b2 = b2 - alpha * db2;
    (w1, b1, w2, b2)
}

pub fn get_predictions(a2: Array2<f64>) -> Array1<f64> {
    // get an float array of the predictions
    let mut predictions = Array::zeros(a2.shape()[0]);
    for i in 0..a2.shape()[0] {
        predictions[i] = a2[(i, a2.slice(s![i, ..]).argmax().unwrap())]
    }
    predictions
}

pub fn get_accuracy(predictions: Array1<f64>, y: Array1<f64>) -> f64 {
    // print(predictions, Y)
    let mut matches = 0.0;
    for i in 0..y.shape()[0] {
        if predictions[i] == y[i] {
            matches += 1.0;
        }
    }
    matches / y.shape()[0] as f64
}

pub fn gradient_descent(
    x: Array1<f64>, y:Array1<f64>, alpha:f64, iterations:usize
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let (w1, b1, w2, b2) = init_params();
    for i in 0..iterations {
        let (z1, a1, z2, a2) = forward_prop(w1, b1, w2, b2, x);
        let (dW1, db1, dW2, db2) = backward_prop(z1, a1, a2, w2, x, y);
        (w1, b1, w2, b2) = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha);
        if i % 10 == 0 {
            println!("Iteration: {:?}", i);
            let predictions = get_predictions(a2);
            println!("{:?}", get_accuracy(predictions, y))
        }
    }
    (w1, b1, w2, b2)
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
