use polars::prelude::*;
use ndarray::prelude::*;
use std::fs::File;


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
    }
}