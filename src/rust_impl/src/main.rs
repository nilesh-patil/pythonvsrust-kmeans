use clap::Parser;
use csv::{Reader, Writer};
use rust_impl::{DataPoint, InitMethod, KMeans};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

/// Command line arguments for K-Means clustering
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CSV file path
    #[arg(long, required = true)]
    input: String,

    /// Output CSV file path
    #[arg(long, required = true)]
    output: String,

    /// Maximum number of clusters to compute
    #[arg(long = "k_clusters_max", required = true)]
    k_clusters_max: usize,

    /// Name of the ID column
    #[arg(long, default_value = "ID")]
    id_column: String,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    random_state: u64,

    /// Maximum iterations per k-means run
    #[arg(long, default_value = "300")]
    max_iterations: usize,

    /// Centroid initialization method: "random" or "k-means++"
    #[arg(long, default_value = "random",
          value_parser = parse_init_method)]
    init: String,
}

fn parse_init_method(s: &str) -> Result<String, String> {
    match s {
        "random" | "k-means++" => Ok(s.to_string()),
        other => Err(format!(
            "unknown init method '{}'; expected 'random' or 'k-means++'",
            other
        )),
    }
}

fn init_method_from_str(s: &str) -> InitMethod {
    match s {
        "k-means++" => InitMethod::KMeansPlusPlus,
        _ => InitMethod::Random,
    }
}

/// Load data from CSV file.
fn load_data(
    filepath: &str,
    id_column: &str,
) -> Result<(Vec<DataPoint>, Vec<String>), Box<dyn Error>> {
    let file = File::open(filepath)?;
    let mut reader = Reader::from_reader(file);

    let headers = reader.headers()?.clone();
    let id_index = headers
        .iter()
        .position(|h| h == id_column)
        .ok_or_else(|| format!("ID column '{}' not found", id_column))?;

    let feature_indices: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != id_index)
        .map(|(i, _)| i)
        .collect();

    let feature_names: Vec<String> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != id_index)
        .map(|(_, name)| name.to_string())
        .collect();

    if feature_indices.is_empty() {
        return Err("No feature columns found (only ID column present)".into());
    }

    let mut data_points = Vec::new();
    for result in reader.records() {
        let record = result?;
        let id = record
            .get(id_index)
            .ok_or("Missing ID value")?
            .to_string();
        let mut features = Vec::new();
        for &idx in &feature_indices {
            let value = record
                .get(idx)
                .ok_or("Missing feature value")?
                .parse::<f64>()
                .map_err(|_| "Failed to parse feature as float")?;
            features.push(value);
        }
        data_points.push(DataPoint { id, features });
    }

    println!(
        "Loaded {} data points with {} features",
        data_points.len(),
        feature_indices.len()
    );

    Ok((data_points, feature_names))
}

/// Save results to CSV file.
fn save_results(
    output_path: &str,
    data: &[DataPoint],
    feature_names: &[String],
    id_column: &str,
    results: &HashMap<usize, Vec<usize>>,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(output_path)?;
    let mut writer = Writer::from_writer(file);

    let mut headers = vec![id_column.to_string()];
    headers.extend(feature_names.iter().cloned());

    let mut k_values: Vec<usize> = results.keys().cloned().collect();
    k_values.sort();
    for k in &k_values {
        headers.push(format!("cluster_{}", k));
    }
    writer.write_record(&headers)?;

    for (i, point) in data.iter().enumerate() {
        let mut row = vec![point.id.clone()];
        for value in &point.features {
            row.push(value.to_string());
        }
        for k in &k_values {
            if let Some(labels) = results.get(k) {
                row.push(labels[i].to_string());
            }
        }
        writer.write_record(&row)?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("K-Means Clustering - Rust Implementation");
    println!("========================================");
    println!("Input file: {}", args.input);
    println!("Output file: {}", args.output);
    println!("Max clusters: {}", args.k_clusters_max);
    println!("ID column: {}", args.id_column);
    println!("Random state: {}", args.random_state);
    println!("Max iterations: {}", args.max_iterations);
    println!("Init method: {}", args.init);
    println!("========================================");

    println!("\nLoading data...");
    let (data, feature_names) = load_data(&args.input, &args.id_column)?;

    let init_method = init_method_from_str(&args.init);
    let mut results = HashMap::new();

    for k in 1..=args.k_clusters_max.min(data.len()) {
        println!("\nRunning k-means with k={}...", k);
        let mut kmeans =
            KMeans::with_init(k, args.max_iterations, args.random_state, init_method.clone());
        kmeans.fit(&data);
        results.insert(k, kmeans.labels);
    }

    println!("\nSaving results...");
    save_results(
        &args.output,
        &data,
        &feature_names,
        &args.id_column,
        &results,
    )?;

    println!("\nResults saved to: {}", args.output);
    println!("Done!");

    Ok(())
}
