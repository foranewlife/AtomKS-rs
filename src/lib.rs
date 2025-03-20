use pyo3::prelude::*;

pub mod atom;


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
// #[pymodule(name = "atomks")]
// fn base(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     Ok(())
// }


#[pymodule]
mod atomks {
    use super::*;

    #[pymodule_export]
    use super::sum_as_string;

    #[pyfunction]
    fn my_sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Arbitrary code to run at the module initialization
        m.add("double2", m.getattr("my_sum_as_string")?)
    }

}

