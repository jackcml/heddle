# heddle

A DSL for image and video transformation.

## Operators

| op | description |
| --- | --- |
| `f(...)` | function; named transform/source with parameters. see below |
| `^k` | speed, where `k` is a float; negative values reverse input |
| `\|` | pipe to compose transforms |
| `over` | composites left and right inputs on top of each other |
| `&`, `/` | horizontal and vertical stacking layout, respectively |
| `>>` | temporal concatenation, left before right. supports transitions |
