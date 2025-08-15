use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting GPU diagnosis...");

    // 1. Create a wgpu instance.
    let instance = wgpu::Instance::default();

    // 2. Enumerate all available adapters (GPUs).
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let mut found_adapter = false;

    println!("\n--- Detected Adapters ---");
    for (i, adapter) in adapters.into_iter().enumerate() {
        found_adapter = true;
        let info = adapter.get_info();
        let limits = adapter.limits();

        println!("\n[Adapter {}]", i);
        println!("  Name: {}", info.name);
        println!("  Backend: {:?}", info.backend);
        println!("  Type: {:?}", info.device_type);
        println!("  Driver: {} ({})", info.driver, info.driver_info);
        
        println!("\n  --- Adapter Limits ---");
        println!("  Max Buffer Size: {} bytes ({} MB)", limits.max_buffer_size, limits.max_buffer_size / (1024 * 1024));
        println!("  Max Storage Buffer Binding Size: {} bytes ({} MB)", limits.max_storage_buffer_binding_size, limits.max_storage_buffer_binding_size / (1024 * 1024));
        // Print all limits for detailed diagnosis
        // println!("  All Limits: {:#?}", limits);
    }

    if !found_adapter {
        println!("\nNo compatible GPU adapters found!");
    }
    println!("\n-----------------------");
    println!("\nDiagnosis finished.");

    Ok(())
}
