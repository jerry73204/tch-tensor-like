# tch-tensor-like: Derive Tensor-like Types for tch-rs

## About this crate

If you are a user of [tch-rs](https://github.com/LaurentMazare/tch-rs), perhaps you ever worked with a complex model input type like this.

```rust
struct ModelInput {
    pub images: Vec<Tensor>,
    pub kind: Tensor,
    pub label: Option<Tensor>,
}
```

Before you feed a batch input of this type into a model, you have to move it to the appropriate device.
It could be tedious to call `tensor.to_device()` for each member of the type.
The `TensorLike` derive macro comes to your rescue.

```rust
use tch_tensor_like::TensorLike;

#[derive(TensorLike)]
struct ModelInput {
    pub images: Vec<Tensor>,
    pub kind: Tensor,
    pub label: Option<Tensor>,
}
```

By deriving the macro, you have `to_device()`, `to_kind()` and `shallow_clone()` out of box.

```rust
let input: ModelInput = fetch_data();
let input = input.to_device(Device::cuda_if_available())
                 .to_kind(Kind::Float)
                 .shallow_clone();
```

For non-tensor members, you can mark the attributes to clone the value instead.

```rust
#[derive(TensorLike)]
struct ModelInput {
    // primitives are copied by default
    pub number: i32,

    // copy the field
    #[tensor_like(copy)]
    pub text: &'static str,

    // clone the field
    #[tensor_like(clone)]
    pub desc: String,
}
```

## Usage

The crate is not published to crates.io yet.
Add the repo link to include this crate in your project.

```toml
[dependencies]
tch-tensor-like = { git = "https://github.com/jerry73204/tch-tensor-like.git", features = ["derive"] }
```

## License

MIT License. See [LICENSE](LICENSE.txt) file.
