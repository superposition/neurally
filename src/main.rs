use neurally::*;
use polars::prelude::*;
use ndarray::prelude::*;

fn main() {
    let result = 2 + 2;
    assert_eq!(result, 4);

    let df = mnist_df().unwrap();
    println!("{:?}", df.head(Some(10)));  
    let data = df.to_ndarray::<Float64Type>().unwrap();
    let m: usize = data.shape()[0] as usize;
    let n: usize = data.shape()[1] as usize;

    let total = 1000;

    println!("{:?}", (m, n));

    let _data_dev = data.slice(s![0..total, ..]).t().to_owned();
    let ydev = data.slice(s![0, ..]).to_owned();
    println!("ydev {:?}", ydev);

    
    let xdev = data.slice(s![1..n, ..]).to_owned();
    println!("xdev {:?}", xdev);

    let data_train = data.slice(s![total..m, ..]).to_owned();
    let data_train = data_train.t().to_owned();
    
    let y_train = data_train.slice(s![0, ..]).to_owned();
    println!("y_train {:?}", y_train);
    let x_train = data_train.slice(s![1..n, ..]).to_owned();
    println!("x_train {:?}", x_train);

    println!("y_one_hot \n{:?}", one_hot(&y_train));

    

}
