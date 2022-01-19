## basic matrix operations  

I needed to perform some statistics for a personal project and I felt the options available where over engineered. 

I'm not working with tensor nor PCA, I just need a simple matrix library for dot multiplication, element wise math operations and some other simple transformations

- This library catch invalid matrix operations at compilation time and is very light weight. 
- Very fast compilation
- All operations are defined to allow to pass a pre allocated matrix. This way avoid unnecessary allocation and deallocation. 
- All operation are defined to mutated the original matrix

The type of operations it does:
- Get matrix diagonal
- Reshape the original matrix if and on if the number of element is equal between the original and new matrixes
- Trim the the original matrix to a new one with less items
- Dot and transpose operations
- Several element wise operation like addition, subtraction, multiplication, division
- Scalar operations on the whole matrix
- And others


## sample test

```rust
    let a = Matrix::<U2, U5>::new();
    let b = Matrix::<U2, U5>::fill(1.0);
    let c = Matrix::<U2, U5>::from_fn(|i| i as f32);

    let mut c1 = c.clone();

    let dia = b.diagonal::<U12, U12>();

    dia.each_i(|row, col, val| {
        if row == col {
            if row < b.len() {
                assert_eq!(val as i32, 1);
            } else {
                assert_eq!(val as i32, 0);
            }
        }
    });

    c1.map_mut(|_, v| v * 2.0);

    let d = c.map_i(|row, col, _| 2.0 * (row + col * c.shape.0) as f32);
    let e = c.map(|_, v| v * v);
    let f = b.transpose();

    let x = Matrix::<U3, U4>::from_fn(|i| i as f32);

    let reshaped = x.reshape::<U12, U1>();

    assert_eq!(
        c1,
        Matrix {
            shape: (2, 5),
            size: 10,
            data: arr![f32; 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        }
    );

    assert_eq!(
        reshaped,
        Matrix {
            shape: (12, 1),
            size: 12,
            data: arr![f32; 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        }
    );

    let trimmed = x.trim::<U2, U4, U2>();

    assert_eq!(
        trimmed,
        Matrix {
            shape: (2, 4),
            size: 8,
            data: arr![f32; 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        }
    );

    assert_eq!(
        a,
        Matrix {
            shape: (2, 5),
            size: 10,
            data: arr![f32; 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    );

    assert_eq!(
        b.data,
        arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    assert_eq!(
        c.data,
        arr![f32; 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );

    assert_eq!(
        d.data,
        arr![f32; 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    );

    assert_eq!(
        e.data,
        arr![f32; 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0]
    );

    assert_eq!(
        f,
        Matrix {
            shape: (5, 2),
            size: 10,
            data: arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    );

    assert_eq!(
        f.dot(&b),
        Matrix {
            shape: (5, 5),
            size: 10,
            data: arr![f32;2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        }
    );

    assert_eq!(
        b.dot(&f),
        Matrix {
            shape: (2, 2),
            size: 4,
            data: arr![f32;5.0, 5.0, 5.0, 5.0]
        }
    );

```