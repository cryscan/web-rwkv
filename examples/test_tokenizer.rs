use anyhow::Result;
use web_rwkv::tokenizer::Tokenizer;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load the specified tokenizer file.
    println!("Loading tokenizer from 'assets/vocab/webrwkv_vocab.json'...");
    let vocab_path = "assets/vocab/webrwkv_vocab.json";
    let content = tokio::fs::read_to_string(vocab_path).await?;
    let tokenizer = Tokenizer::new(&content)?;
    println!("Tokenizer loaded successfully.\n");

    // --- ENCODING TEST ---
    // The tokenizer expects a byte slice, so we create a string of
    // space-separated tokens as our "sentence".
    let original_text = "semantic_15 semantic_1023 semantic_511";
    println!("--- Encoding Test ---");
    println!("Original text: '{}'", original_text);

    let encoded_ids = tokenizer.encode(original_text.as_bytes())?;
    println!("Encoded IDs: {:?}", encoded_ids);
    println!("---------------------\n");

    // --- DECODING TEST ---
    let original_ids: Vec<u32> = vec![15, 1023, 511];
    println!("--- Decoding Test ---");
    println!("Original IDs: {:?}", original_ids);

    let decoded_bytes = tokenizer.decode(&original_ids)?;
    // The result is a byte vector, convert it to a string for printing.
    let decoded_text = String::from_utf8_lossy(&decoded_bytes);
    println!("Decoded text: '{}'", decoded_text);
    println!("---------------------\n");

    // --- ROUND-TRIP TEST ---
    println!("--- Round-Trip Test ---");
    println!("Original text: '{}'", original_text);
    let round_trip_bytes = tokenizer.decode(&encoded_ids)?;
    let round_trip_text = String::from_utf8_lossy(&round_trip_bytes);
    println!("Encoded and then decoded text: '{}'", round_trip_text);

    assert_eq!(original_text, round_trip_text);
    println!("\nSuccess! The decoded text matches the original text.");
    println!("-------------------------");

    Ok(())
}