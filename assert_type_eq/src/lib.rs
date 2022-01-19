// use typenum::Same;

#[macro_export]
macro_rules! assert_type_eq {
    ($a:ty, $b:ty) => {
        let _: <$a as Same<$b>>::Output;
    };
}

