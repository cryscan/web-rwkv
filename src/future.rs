#[cfg(not(target_arch = "wasm32"))]
pub trait Future<Output>: std::future::Future<Output = Output> + Send {}

#[cfg(not(target_arch = "wasm32"))]
impl<Output, T: std::future::Future<Output = Output> + Send> Future<Output> for T {}

#[cfg(target_arch = "wasm32")]
pub trait Future<Output>: std::future::Future<Output = Output> {}

#[cfg(target_arch = "wasm32")]
impl<Output, T: std::future::Future<Output = Output>> Future<Output> for T {}
