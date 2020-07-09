use approx::assert_abs_diff_eq;
use std::collections::{BTreeMap, HashMap, LinkedList, VecDeque};
use tch::{kind::FLOAT_CPU, Device, Kind, Tensor};
use tch_tensor_like::TensorLike;

#[test]
fn shallow_clone_test() {
    let maybe_cuda = Device::cuda_if_available();

    let from = (0..1000)
        .map(|_| Tensor::randn(&[], FLOAT_CPU))
        .collect::<Vec<_>>();

    let to = from
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone();

    from.into_iter().zip(to.into_iter()).for_each(|(lhs, rhs)| {
        assert_eq!(lhs.device(), Device::Cpu);
        assert_eq!(lhs.kind().unwrap(), Kind::Float);

        assert_eq!(rhs.device(), maybe_cuda);
        assert_eq!(rhs.kind().unwrap(), Kind::Double);

        assert_abs_diff_eq!(f64::from(lhs), f64::from(rhs));
    });
}

#[test]
fn collections_test() {
    let maybe_cuda = Device::cuda_if_available();

    let iter = (0..16).map(|_| Tensor::randn(&[], FLOAT_CPU));

    // vec
    iter.clone()
        .collect::<Vec<_>>()
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone()
        .into_iter()
        .for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });

    // linked list
    iter.clone()
        .collect::<LinkedList<_>>()
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone()
        .into_iter()
        .for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });

    // vec deque
    iter.clone()
        .collect::<VecDeque<_>>()
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone()
        .into_iter()
        .for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });

    let pair_iter = (0..16).map(|index| (index, Tensor::randn(&[], FLOAT_CPU)));

    // hash map
    pair_iter
        .clone()
        .collect::<HashMap<_, _>>()
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone()
        .into_iter()
        .for_each(|(_, tensor)| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });

    // btree map
    pair_iter
        .clone()
        .collect::<BTreeMap<_, _>>()
        .to_device(maybe_cuda)
        .to_kind(Kind::Double)
        .shallow_clone()
        .into_iter()
        .for_each(|(_, tensor)| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });
}

#[test]
#[cfg(feature = "derive")]
fn derive_test() {
    let maybe_cuda = Device::cuda_if_available();

    // unit struct
    #[derive(TensorLike)]
    struct Unit;

    {
        let unit = Unit;
        unit.to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();
    }

    // empty tuple struct
    #[derive(TensorLike)]
    struct EmptyTuple();

    {
        let tup = EmptyTuple();
        tup.to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();
    }

    // tuple struct
    #[derive(TensorLike)]
    struct Tuple(u8, i16, f32, Tensor, Vec<Tensor>);

    {
        let from = Tuple(
            0,
            -1,
            3.14,
            Tensor::randn(&[], FLOAT_CPU),
            vec![Tensor::randn(&[], FLOAT_CPU), Tensor::randn(&[], FLOAT_CPU)],
        );
        let to = from
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();

        assert_eq!(to.3.device(), maybe_cuda);
        assert_eq!(to.3.kind().unwrap(), Kind::Double);

        to.4.iter().for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });
    }

    // empty struct
    #[derive(TensorLike)]
    struct EmptyNamed {}

    {
        let named = EmptyNamed {};
        named
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();
    }

    // named struct
    #[derive(TensorLike)]
    struct Named {
        a: u8,
        b: i16,
        c: f32,
        d: Tensor,
        e: Vec<Tensor>,
    }

    {
        let from = Named {
            a: 0,
            b: -1,
            c: 3.14,
            d: Tensor::randn(&[], FLOAT_CPU),
            e: vec![Tensor::randn(&[], FLOAT_CPU), Tensor::randn(&[], FLOAT_CPU)],
        };
        let to = from
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();

        assert_eq!(to.d.device(), maybe_cuda);
        assert_eq!(to.d.kind().unwrap(), Kind::Double);

        to.e.iter().for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });
    }
}
