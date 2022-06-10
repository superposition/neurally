use rand::{Rng, thread_rng};
use polars::prelude::*;
use ndarray::prelude::*;
use std::fs::File;
use std::sync::Arc;


fn mnist_df() -> Result<DataFrame> {
    let file = File::open("mnist.csv")
                    .expect("could not read file");

    CsvReader::new(file)
            .infer_schema(None)
            .has_header(true)
            .finish()
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
