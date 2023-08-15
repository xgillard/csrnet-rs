use burn_import::onnx::ModelGen;
use structopt::StructOpt;

/// Parses an onnx model and creates the accompanying burn-rs model.
#[derive(Debug, structopt::StructOpt)]
struct Args {
    /// The path to onxx model
    #[structopt(short, long, default_value="./resources/csrnet.onnx")]
    input: String,
    /// The path where to output everything
    #[structopt(short, long, default_value="./csrnet-infer/src/model")]
    out_dir: String,
}

fn main() {
    let args = Args::from_args();
    // Generate the model code and state file from the ONNX file.
    ModelGen::new()
        .development(false)
        .input(&args.input)   // Path to the ONNX model
        .out_dir(&args.out_dir)               // Directory for the generated Rust source file (under target/)
        .run_from_cli();
}