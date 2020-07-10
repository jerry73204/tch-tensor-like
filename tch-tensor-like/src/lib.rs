#[cfg(feature = "derive")]
pub use tch_tensor_like_derive::TensorLike;

use std::{
    collections::{BTreeMap, HashMap, LinkedList, VecDeque},
    hash::Hash,
};
use tch::{Device, Kind, TchError, Tensor};

pub trait TensorLike
where
    Self: Sized,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError>;
    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError>;
    fn shallow_clone(&self) -> Self;

    fn to_device(&self, device: Device) -> Self {
        self.f_to_device(device).unwrap()
    }

    fn to_kind(&self, kind: Kind) -> Self {
        self.f_to_kind(kind).unwrap()
    }
}

// primitives

macro_rules! impl_for_primitive {
    ($name:ty) => {
        impl TensorLike for $name {
            fn f_to_device(&self, _device: Device) -> Result<Self, TchError> {
                Ok(*self)
            }

            fn f_to_kind(&self, _kind: Kind) -> Result<Self, TchError> {
                Ok(*self)
            }

            fn shallow_clone(&self) -> Self {
                *self
            }
        }
    };
}

impl_for_primitive!(bool);
impl_for_primitive!(f32);
impl_for_primitive!(f64);
impl_for_primitive!(usize);
impl_for_primitive!(u8);
impl_for_primitive!(u16);
impl_for_primitive!(u32);
impl_for_primitive!(u64);
impl_for_primitive!(u128);
impl_for_primitive!(isize);
impl_for_primitive!(i8);
impl_for_primitive!(i16);
impl_for_primitive!(i32);
impl_for_primitive!(i64);
impl_for_primitive!(i128);

// reference

impl<T> TensorLike for &T {
    fn f_to_device(&self, _device: Device) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn f_to_kind(&self, _kind: Kind) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn shallow_clone(&self) -> Self {
        *self
    }
}

// pointer

impl<T> TensorLike for *const T {
    fn f_to_device(&self, _device: Device) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn f_to_kind(&self, _kind: Kind) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn shallow_clone(&self) -> Self {
        *self
    }
}

impl<T> TensorLike for *mut T {
    fn f_to_device(&self, _device: Device) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn f_to_kind(&self, _kind: Kind) -> Result<Self, TchError> {
        Ok(*self)
    }

    fn shallow_clone(&self) -> Self {
        *self
    }
}

// tuples

impl<T1> TensorLike for (T1,)
where
    T1: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        Ok((self.0.f_to_device(device)?,))
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        Ok((self.0.f_to_kind(kind)?,))
    }

    fn shallow_clone(&self) -> Self {
        (self.0.shallow_clone(),)
    }
}

impl<T1, T2> TensorLike for (T1, T2)
where
    T1: TensorLike,
    T2: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        Ok((self.0.f_to_device(device)?, self.1.f_to_device(device)?))
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        Ok((self.0.f_to_kind(kind)?, self.1.f_to_kind(kind)?))
    }

    fn shallow_clone(&self) -> Self {
        (self.0.shallow_clone(), self.1.shallow_clone())
    }
}

impl<T1, T2, T3> TensorLike for (T1, T2, T3)
where
    T1: TensorLike,
    T2: TensorLike,
    T3: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_device(device)?,
            self.1.f_to_device(device)?,
            self.2.f_to_device(device)?,
        ))
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_kind(kind)?,
            self.1.f_to_kind(kind)?,
            self.2.f_to_kind(kind)?,
        ))
    }

    fn shallow_clone(&self) -> Self {
        (
            self.0.shallow_clone(),
            self.1.shallow_clone(),
            self.2.shallow_clone(),
        )
    }
}

impl<T1, T2, T3, T4> TensorLike for (T1, T2, T3, T4)
where
    T1: TensorLike,
    T2: TensorLike,
    T3: TensorLike,
    T4: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_device(device)?,
            self.1.f_to_device(device)?,
            self.2.f_to_device(device)?,
            self.3.f_to_device(device)?,
        ))
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_kind(kind)?,
            self.1.f_to_kind(kind)?,
            self.2.f_to_kind(kind)?,
            self.3.f_to_kind(kind)?,
        ))
    }

    fn shallow_clone(&self) -> Self {
        (
            self.0.shallow_clone(),
            self.1.shallow_clone(),
            self.2.shallow_clone(),
            self.3.shallow_clone(),
        )
    }
}

impl<T1, T2, T3, T4, T5> TensorLike for (T1, T2, T3, T4, T5)
where
    T1: TensorLike,
    T2: TensorLike,
    T3: TensorLike,
    T4: TensorLike,
    T5: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_device(device)?,
            self.1.f_to_device(device)?,
            self.2.f_to_device(device)?,
            self.3.f_to_device(device)?,
            self.4.f_to_device(device)?,
        ))
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        Ok((
            self.0.f_to_kind(kind)?,
            self.1.f_to_kind(kind)?,
            self.2.f_to_kind(kind)?,
            self.3.f_to_kind(kind)?,
            self.4.f_to_kind(kind)?,
        ))
    }

    fn shallow_clone(&self) -> Self {
        (
            self.0.shallow_clone(),
            self.1.shallow_clone(),
            self.2.shallow_clone(),
            self.3.shallow_clone(),
            self.4.shallow_clone(),
        )
    }
}

// tensor

impl TensorLike for Tensor {
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.f_to_device(device)
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.f_to_kind(kind)
    }

    fn shallow_clone(&self) -> Self {
        self.shallow_clone()
    }
}

// collections

impl<T> TensorLike for Vec<T>
where
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.iter()
            .map(|tensor| tensor.f_to_device(device))
            .collect()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.iter().map(|tensor| tensor.f_to_kind(kind)).collect()
    }

    fn shallow_clone(&self) -> Self {
        self.iter().map(|tensor| tensor.shallow_clone()).collect()
    }
}

impl<T> TensorLike for LinkedList<T>
where
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.iter()
            .map(|tensor| tensor.f_to_device(device))
            .collect()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.iter().map(|tensor| tensor.f_to_kind(kind)).collect()
    }

    fn shallow_clone(&self) -> Self {
        self.iter().map(|tensor| tensor.shallow_clone()).collect()
    }
}

impl<T> TensorLike for VecDeque<T>
where
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.iter()
            .map(|tensor| tensor.f_to_device(device))
            .collect()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.iter().map(|tensor| tensor.f_to_kind(kind)).collect()
    }

    fn shallow_clone(&self) -> Self {
        self.iter().map(|tensor| tensor.shallow_clone()).collect()
    }
}

impl<K, T> TensorLike for HashMap<K, T>
where
    K: Eq + Hash + Clone,
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.iter()
            .map(|(key, tensor)| Ok((key.clone(), tensor.f_to_device(device)?)))
            .collect()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.iter()
            .map(|(key, tensor)| Ok((key.clone(), tensor.f_to_kind(kind)?)))
            .collect()
    }

    fn shallow_clone(&self) -> Self {
        self.iter()
            .map(|(key, tensor)| (key.clone(), tensor.shallow_clone()))
            .collect()
    }
}

impl<K, T> TensorLike for BTreeMap<K, T>
where
    K: Ord + Clone,
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.iter()
            .map(|(key, tensor)| Ok((key.clone(), tensor.f_to_device(device)?)))
            .collect()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.iter()
            .map(|(key, tensor)| Ok((key.clone(), tensor.f_to_kind(kind)?)))
            .collect()
    }

    fn shallow_clone(&self) -> Self {
        self.iter()
            .map(|(key, tensor)| (key.clone(), tensor.shallow_clone()))
            .collect()
    }
}

// option

impl<T> TensorLike for Option<T>
where
    T: TensorLike,
{
    fn f_to_device(&self, device: Device) -> Result<Self, TchError> {
        self.as_ref()
            .map(|tensor| tensor.f_to_device(device))
            .transpose()
    }

    fn f_to_kind(&self, kind: Kind) -> Result<Self, TchError> {
        self.as_ref()
            .map(|tensor| tensor.f_to_kind(kind))
            .transpose()
    }

    fn shallow_clone(&self) -> Self {
        self.as_ref().map(|tensor| tensor.shallow_clone())
    }
}
