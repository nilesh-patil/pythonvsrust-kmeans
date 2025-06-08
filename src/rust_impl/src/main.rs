use clap::Parser;
use csv::{Reader, Writer};
use rand::prelude::*;
// use serde::{Deserialize, Serialize}; // Not needed for current implementation
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
}

/// Represents a data point with features
#[derive(Debug, Clone)]
struct DataPoint {
    id: String,
    features: Vec<f64>,
}

/// K-Means clustering implementation
struct KMeans {
    k: usize,
    max_iterations: usize,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    rng: StdRng,
}

impl KMeans {
    fn new(k: usize, max_iterations: usize, seed: u64) -> Self {
        KMeans {
            k,
            max_iterations,
            centroids: Vec::new(),
            labels: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Initialize centroids by randomly selecting k data points
    fn initialize_centroids(&mut self, data: &[DataPoint]) {
        if self.k >= data.len() {
            // If k >= n_samples, each point is its own cluster
            self.centroids = data.iter().map(|p| p.features.clone()).collect();
            self.k = data.len();
        } else {
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.shuffle(&mut self.rng);
            self.centroids = indices[..self.k]
                .iter()
                .map(|&i| data[i].features.clone())
                .collect();
        }
    }

    /// Assign each point to the nearest centroid
    fn assign_clusters(&mut self, data: &[DataPoint]) {
        self.labels = data
            .iter()
            .map(|point| {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for (cluster_idx, centroid) in self.centroids.iter().enumerate() {
                    let distance = Self::euclidean_distance(&point.features, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = cluster_idx;
                    }
                }
                best_cluster
            })
            .collect();
    }

    /// Update centroids based on assigned points
    fn update_centroids(&mut self, data: &[DataPoint]) -> bool {
        let old_centroids = self.centroids.clone();
        
        for cluster_idx in 0..self.k {
            let cluster_points: Vec<&DataPoint> = data
                .iter()
                .zip(self.labels.iter())
                .filter(|(_, &label)| label == cluster_idx)
                .map(|(point, _)| point)
                .collect();

            if !cluster_points.is_empty() {
                // Calculate mean of all points in the cluster
                let n_features = data[0].features.len();
                let mut new_centroid = vec![0.0; n_features];
                
                for point in &cluster_points {
                    for (i, &value) in point.features.iter().enumerate() {
                        new_centroid[i] += value;
                    }
                }
                
                let count = cluster_points.len() as f64;
                for value in &mut new_centroid {
                    *value /= count;
                }
                
                self.centroids[cluster_idx] = new_centroid;
            } else {
                // If cluster is empty, reinitialize with a random point
                let random_idx = self.rng.gen_range(0..data.len());
                self.centroids[cluster_idx] = data[random_idx].features.clone();
            }
        }

        // Check if centroids have converged
        old_centroids == self.centroids
    }

    /// Fit the K-Means model to the data
    fn fit(&mut self, data: &[DataPoint]) {
        if data.is_empty() {
            return;
        }

        // Handle edge case where k=1
        if self.k == 1 {
            self.labels = vec![0; data.len()];
            self.centroids = vec![self.calculate_mean(data)];
            return;
        }

        // Initialize centroids
        self.initialize_centroids(data);

        // Handle edge case where k >= n_samples
        if self.k >= data.len() {
            self.labels = (0..data.len()).collect();
            return;
        }

        // Iterative optimization
        for _iteration in 0..self.max_iterations {
            let old_labels = self.labels.clone();
            
            // Assign clusters
            self.assign_clusters(data);
            
            // Check for convergence (labels haven't changed)
            if !old_labels.is_empty() && old_labels == self.labels {
                break;
            }
            
            // Update centroids
            let converged = self.update_centroids(data);
            if converged {
                break;
            }
        }
    }

    /// Calculate mean of all data points (used for k=1)
    fn calculate_mean(&self, data: &[DataPoint]) -> Vec<f64> {
        let n_features = data[0].features.len();
        let mut mean = vec![0.0; n_features];
        
        for point in data {
            for (i, &value) in point.features.iter().enumerate() {
                mean[i] += value;
            }
        }
        
        let count = data.len() as f64;
        for value in &mut mean {
            *value /= count;
        }
        
        mean
    }
}

/// Load data from CSV file
fn load_data(filepath: &str, id_column: &str) -> Result<(Vec<DataPoint>, Vec<String>), Box<dyn Error>> {
    let file = File::open(filepath)?;
    let mut reader = Reader::from_reader(file);
    
    // Get headers
    let headers = reader.headers()?.clone();
    let id_index = headers
        .iter()
        .position(|h| h == id_column)
        .ok_or_else(|| format!("ID column '{}' not found", id_column))?;
    
    // Get feature column indices
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
    
    // Read data
    let mut data_points = Vec::new();
    
    for result in reader.records() {
        let record = result?;
        
        let id = record.get(id_index)
            .ok_or("Missing ID value")?
            .to_string();
        
        let mut features = Vec::new();
        for &idx in &feature_indices {
            let value = record.get(idx)
                .ok_or("Missing feature value")?
                .parse::<f64>()
                .map_err(|_| "Failed to parse feature as float")?;
            features.push(value);
        }
        
        data_points.push(DataPoint { id, features });
    }
    
    println!("Loaded {} data points with {} features", data_points.len(), feature_indices.len());
    
    Ok((data_points, feature_names))
}

/// Save results to CSV file
fn save_results(
    output_path: &str,
    data: &[DataPoint],
    feature_names: &[String],
    id_column: &str,
    results: &HashMap<usize, Vec<usize>>,
) -> Result<(), Box<dyn Error>> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(output_path)?;
    let mut writer = Writer::from_writer(file);
    
    // Write headers
    let mut headers = vec![id_column.to_string()];
    headers.extend(feature_names.iter().cloned());
    
    // Add cluster columns
    let mut k_values: Vec<usize> = results.keys().cloned().collect();
    k_values.sort();
    
    for k in &k_values {
        headers.push(format!("cluster_{}", k));
    }
    
    writer.write_record(&headers)?;
    
    // Write data rows
    for (i, point) in data.iter().enumerate() {
        let mut row = vec![point.id.clone()];
        
        // Add features
        for value in &point.features {
            row.push(value.to_string());
        }
        
        // Add cluster assignments
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
    println!("========================================");
    
    // Load data
    println!("\nLoading data...");
    let (data, feature_names) = load_data(&args.input, &args.id_column)?;
    
    // Run k-means for multiple k values
    let mut results = HashMap::new();
    
    for k in 1..=args.k_clusters_max.min(data.len()) {
        println!("\nRunning k-means with k={}...", k);
        
        let mut kmeans = KMeans::new(k, args.max_iterations, args.random_state);
        kmeans.fit(&data);
        
        results.insert(k, kmeans.labels);
    }
    
    // Save results
    println!("\nSaving results...");
    save_results(&args.output, &data, &feature_names, &args.id_column, &results)?;
    
    println!("\nResults saved to: {}", args.output);
    println!("Done!");
    
    Ok(())
}
