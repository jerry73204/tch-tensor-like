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
fn derive_types_test() {
    let maybe_cuda = Device::cuda_if_available();

    // unit struct
    {
        #[derive(Debug, TensorLike, PartialEq, Eq)]
        struct Unit;

        let unit = Unit;
        assert_eq!(
            unit.to_device(maybe_cuda)
                .to_kind(Kind::Double)
                .shallow_clone(),
            Unit
        );
    }

    // empty tuple struct
    {
        #[derive(TensorLike)]
        struct EmptyTuple();

        let tup = EmptyTuple();
        tup.to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();
    }

    // tuple struct
    {
        #[derive(TensorLike)]
        struct Tuple(u8, i16, f32, Tensor, Vec<Tensor>);

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

        assert_eq!(to.0, 0);
        assert_eq!(to.1, -1);
        assert_eq!(to.2, 3.14);
        assert_eq!(to.3.device(), maybe_cuda);
        assert_eq!(to.3.kind().unwrap(), Kind::Double);

        to.4.iter().for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });
    }

    // empty struct
    {
        #[derive(TensorLike)]
        struct EmptyNamed {}

        let named = EmptyNamed {};
        named
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();
    }

    // named struct
    {
        #[derive(TensorLike)]
        struct Named {
            a: u8,
            b: i16,
            c: f32,
            d: Tensor,
            e: Vec<Tensor>,
        }

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

        assert_eq!(to.a, 0);
        assert_eq!(to.b, -1);
        assert_eq!(to.c, 3.14);
        assert_eq!(to.d.device(), maybe_cuda);
        assert_eq!(to.d.kind().unwrap(), Kind::Double);

        to.e.iter().for_each(|tensor| {
            assert_eq!(tensor.device(), maybe_cuda);
            assert_eq!(tensor.kind().unwrap(), Kind::Double);
        });
    }

    // enum
    {
        #[derive(TensorLike)]
        enum Enum {
            Unit,
            Tuple(u8, i16, f32, Tensor, Vec<Tensor>),
            Named {
                a: u8,
                b: i16,
                c: f32,
                d: Tensor,
                e: Vec<Tensor>,
            },
        }

        // unit variant
        {
            let unit = Enum::Unit;
            match unit
                .to_device(maybe_cuda)
                .to_kind(Kind::Double)
                .shallow_clone()
            {
                Enum::Unit => (),
                _ => unreachable!(),
            }
        }

        // tuple variant
        {
            let from = Enum::Tuple(
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

            match to {
                Enum::Tuple(a, b, c, tensor, vec) => {
                    assert_eq!(a, 0);
                    assert_eq!(b, -1);
                    assert_eq!(c, 3.14);

                    assert_eq!(tensor.device(), maybe_cuda);
                    assert_eq!(tensor.kind().unwrap(), Kind::Double);

                    vec.iter().for_each(|tensor| {
                        assert_eq!(tensor.device(), maybe_cuda);
                        assert_eq!(tensor.kind().unwrap(), Kind::Double);
                    });
                }
                _ => unreachable!(),
            }
        }

        // named variant
        {
            let from = Enum::Named {
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

            match to {
                Enum::Named { a, b, c, d, e } => {
                    assert_eq!(a, 0);
                    assert_eq!(b, -1);
                    assert_eq!(c, 3.14);

                    assert_eq!(d.device(), maybe_cuda);
                    assert_eq!(d.kind().unwrap(), Kind::Double);

                    e.iter().for_each(|tensor| {
                        assert_eq!(tensor.device(), maybe_cuda);
                        assert_eq!(tensor.kind().unwrap(), Kind::Double);
                    });
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
#[cfg(feature = "derive")]
fn derive_clone_test() {
    let maybe_cuda = Device::cuda_if_available();

    // tuple struct
    {
        #[derive(TensorLike)]
        struct Tuple(
            Tensor,
            #[tensor_like(copy)] &'static str,
            #[tensor_like(clone)] String,
        );

        let from = Tuple(Tensor::randn(&[], FLOAT_CPU), "mighty", "tch".into());
        let to = from
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();

        assert_eq!(to.0.device(), maybe_cuda);
        assert_eq!(to.0.kind().unwrap(), Kind::Double);
        assert_eq!(to.1, "mighty");
        assert_eq!(to.2, "tch".to_string());
    }

    // named struct
    {
        #[derive(TensorLike)]
        struct Named {
            a: Tensor,
            #[tensor_like(copy)]
            b: &'static str,
            #[tensor_like(clone)]
            c: String,
        }

        let from = Named {
            a: Tensor::randn(&[], FLOAT_CPU),
            b: "mighty",
            c: "tch".into(),
        };
        let to = from
            .to_device(maybe_cuda)
            .to_kind(Kind::Double)
            .shallow_clone();

        assert_eq!(to.a.device(), maybe_cuda);
        assert_eq!(to.a.kind().unwrap(), Kind::Double);
        assert_eq!(to.b, "mighty");
        assert_eq!(to.c, "tch".to_string());
    }
}
