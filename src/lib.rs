// use rand::{thread_rng};
use polars::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
// use ndarray_rand::rand::{Rng, RngCore};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use std::fs::File;


pub fn mnist_df() -> Result<DataFrame> {
    let file = File::open("mnist.csv")
                    .expect("could not read file");

    CsvReader::new(file)
            .infer_schema(None)
            .has_header(true)
            .finish()
}

pub fn init_params() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) { // correct
    let normal = Normal::new(0.0, 1.0).unwrap();

    let w1 = Array::random((10, 784), normal);
    let b1 = Array::random(10, normal);
    let w2 = Array::random((10, 10), normal);
    let b2 = Array::random(10, normal);
    
    (w1, b1, w2, b2)
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let e: f64 = 2.718281828459;
    let exp = x.mapv(|n| e.powf(n));
    &exp / exp.sum()
}


pub fn forward_prop(
    w1: &Array2<f64>, b1: &Array1<f64>, w2: &Array2<f64>, b2: &Array1<f64>, x: &Array2<f64>
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) { // correct
    let z1 = w1.dot(x) + b1;
    let a1 = z1.mapv(|x| relu(x));
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(&z2);

    (z1, a1, z2, a2)
}

pub fn one_hot(y: &Array1<f64>) -> Array2<f64> {
    let mut y_one_hot = Array::zeros((y.shape()[0], 10));
    for i in 0..y.len() {
        y_one_hot[(i, y[i] as usize)] = 1.0;
    }
    y_one_hot.t().to_owned()

}

pub fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|n| if n < 0.0 { 0.0 } else { n })
}

// backprop 
pub fn backward_prop(
    z1: &Array2<f64>, 
    a1: &Array2<f64>, 
    a2: &Array2<f64>,  
    w2: &Array2<f64>,
    x: &Array2<f64>, 
    y: &Array1<f64>
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let m = a2.shape()[0] as f64;
    let one_hot_y = one_hot(y);
    let dz2 = a2 - &one_hot_y;
    let dw2 = dz2.dot(&a1.t()) * (1.0 / m);
    let db2 = 1.0 / m * dz2.sum_axis(Axis(0));  
    let dz1 = w2.t().dot(&dz2) * &relu_derivative(z1);
    let dw1 = 1.0 / m * dz1.dot(&x.t()); 
    let db1 = (1.0 / m) * dz1.sum_axis(Axis(0));

    (dw1, db1, dw2, db2)
}


pub fn update_params(
    w1: &Array2<f64>, 
    b1: &Array1<f64>, 
    w2: &Array2<f64>, 
    b2: &Array1<f64>, 
    dw1: &Array2<f64>, 
    db1: &Array1<f64>, 
    dw2: &Array2<f64> , 
    db2: &Array1<f64>, 
    alpha: f64
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let w1 = w1 - alpha * dw1;
    let b1 = b1 - alpha * db1;
    let w2 = w2 - alpha * dw2;
    let b2 = b2 - alpha * db2;
    (w1, b1, w2, b2)
}



pub fn get_predictions(a2: Array2<f64>) -> Array1<f64> {
    //get an float array of the predictions
    let mut predictions = Array::zeros(a2.shape()[0]);

    for i in 0..a2.shape()[0] {
        let index_max = a2.slice(s![i, ..]).argmax().unwrap();
        let col_max = a2[[i, index_max]];
        predictions[i] = col_max;
    }
    predictions
}

pub fn get_accuracy(predictions: Array1<f64>, y: &Array1<f64>) -> f64 {
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
    x: Array2<f64>, y: Array1<f64>, alpha:f64, iterations:usize
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let (mut w1, mut b1, mut w2, mut b2) = init_params();
    for i in 0..iterations {
        let (z1, a1, _z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &x);
        let (dw1, db1, dw2, db2) = backward_prop(&z1, &a1, &a2, &w2, &x, &y);
        (w1, b1, w2, b2) = update_params(&w1, &b1, &w2, &b2, &dw1, &db1, &dw2, &db2, alpha);
        if i % 10 == 0 {
            println!("Iteration: {:?}", i);
            let predictions = get_predictions(a2);
            println!("{:?}", get_accuracy(predictions, &y))
        }
    }
    (w1, b1, w2, b2)
}
