had to define this crate with only on macro inse typenum > 1.11.2 changed the `assert_type_eq` implementation.

current version of typemun now gives `can't use generic parameters from outer function` if the `assert_type_eq` is used inside a function

however the `assert_type_eq` from version 1.11.2 still works