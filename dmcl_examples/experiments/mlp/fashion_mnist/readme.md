* setting1: main Gibbs approach, 3000 batch size, original data
* setting4: as in setting 1, but with transformed data using
```
transforms.RandomRotation(degrees=(-30, 30))
```
* setting5: as in setting 1, but with transformed data using
```
transforms.RandomApply(
    torch.nn.ModuleList([
        transforms.GaussianBlur(kernel_size=(9, 9), sigma=(1., 1.5)),
    ]),
    0.9
)
```
* setting6: as in setting 1, but with transformed data using
```
transforms.RandomInvert(p=0.5)
```
* setting7: as in setting 1, but with batch size of 1800
* setting8: as in setting 1, but with batch size of 4200
* setting9: as in setting 1, but with batch size of 600
