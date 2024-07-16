use std::net::SocketAddr;
use std::path::PathBuf;

use askama::Template;
use axum::*;
use axum::body::Body;
use axum::extract::{State, Path};
use axum::http::Response;
use axum::routing::{get, post};
use structopt::StructOpt;
use std::fs::read_dir;

#[derive(Debug, StructOpt)]
struct Args {
    /// the path to the dataset to create
    pub dataset: String,
}

#[tokio::main]
async fn main() {
    let Args { dataset } = Args::from_args();
    let path = PathBuf::from(&dataset).join("images");
    let entries = read_dir(&path).expect("could not read images dataset directory");
    let mut images = vec![];
    for entry in entries {
        if let Ok(entry) = entry {
            images.push(entry.path().to_string_lossy().to_string())
        }
    }
    
    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/:id",        get(root))
        .route("/label/:id",  post(labelize))
        .route("/image/:id",  get(image))
        .with_state((dataset, images));

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    let addr = SocketAddr::from(([127, 0, 0, 1], 8888));
    let server = axum::Server::bind(&addr)
        .serve(app.into_make_service());
    //
    open::that("http://localhost:8888/0").unwrap();
    server.await.unwrap();
}

#[derive(Debug, Clone, Template, serde::Serialize, serde::Deserialize)]
#[template(path="labeling.html")]
struct Labeling {
    id: usize, 
    img: String,
    dataset: String,
    done: bool,
}

async fn root(
    Path(id): Path<usize>,
    State((dataset, images)): State<(String, Vec<String>)>
) -> Labeling {
    if id >= images.len() {
        Labeling{id, dataset: dataset.replace("\\", "/"), img: String::new(), done: true}
    } else {
        Labeling{id, dataset: dataset.replace("\\", "/"), img: images[id].replace("\\", "/"), done: false}   
    }
}

async fn image(
    Path(id): Path<usize>, 
    State((_dataset, images)): State<(String, Vec<String>)>) 
-> Response<Body>
{
    let data = std::fs::read(&images[id])
        .expect("could not read image file");
    let response = Response::builder()
        .header("Content-Type", "image/jpg")
        .body(Body::from(data))
        .expect("could not build response");
    response
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LabelingRequest {
    dataset: String,
    img:     String,
    positions: Vec<[f32;2]>
}
async fn labelize(
    Json(LabelingRequest{img, dataset, positions}): Json<LabelingRequest>
) {
    let name = std::path::PathBuf::from(&img);
    let name = name.file_stem().unwrap().to_string_lossy().to_string();
    let path = PathBuf::from(&dataset).join("ground_truth").join(format!("{name}.npy"));
    let data = ndarray::Array2::from(positions);
    
    ndarray_npy::write_npy(
        path,
        &data
    ).ok();
}