use neurally::mnist_df;
fn main() {
    let df = mnist_df().unwrap();
    println!("{:?}", df.head(Some(10)));  
    
    //load into ndarray
    //mnist_df
    let data = df.to_ndarray::<UInt64Type>().unwrap();
    //println!("{:?}", data.shape());

    //covert to u8
    let data = data.mapv(|x| x as u8);
    //println!("{:?}", data.shape());

    //transpose data
    let data = data.t();

    let m: usize = data.shape()[0] as usize;
    let n: usize = data.shape()[1] as usize;

    let total = 1000;

    println!("{:?}", (m, n));

    let data_dev = data.slice(s![0..total, 1]);

    let ydev = data.slice(s![0, 1]);
    print!("{:?}", ydev);

    
    let xdev = data.slice(s![1..n, 1]);
    print!("{:?}", xdev);

    let data_train = data.slice(s![total..m, 1]);

    let y_train = data_train.slice(s![0]);
    print!("{:?}", y_train);
    let x_train = data_train.slice(s![1..n]);
    print!("{:?}", x_train);
}
