# heddle

A DSL for image and video transformation.

<img src="examples/outputs/wwwwaiting.gif" alt="animated gif result" />

`wwwwait^0.2 >> (wwwwait^0.2 | hflip) > "wwwwaiting.gif"`

- Currently implemented: tree-walk interpreter with the base operators and named functions below, backed by `pillow`
- Future:
  - More comprehensive handling of number types (`ms`, `s`, `%`)
  - Aliasing / assign names to pipelines
  - Custom functions defined by GLSL shaders

## Sources

Input images and videos are referenced with automatically defined variables. For example, when `heddle` is run in a directory with `im.gif`, `im^2 | hflip` creates a copy at 2x speed, flipped horizontally.

## Operators

| op | description |
| --- | --- |
| `f(...)` | function; named transform/source with parameters. see below |
| `[t0:t1, y0:y1, x0:x1]` | slice (time, y, x) |
| `^k` | speed, where `k` is a float; negative values reverse input |
| `\|` | pipe to compose transforms |
| `over` | composites left and right inputs on top of each other |
| `&`, `/` | horizontal and vertical stacking layout, respectively |
| `>>` | temporal concatenation, left before right. supports transitions |

## Functions

Functions are used for transformations not covered by the operators and for transition markers used inside a timeline.

| fn | description |
| --- | --- |
| `text(str, pos)` | displays the given `str` in white over the input, aligned according to `pos` |
| `blur(stdev)` | applies a Gaussian blur |
| `scale(factor)` | scales input by single factor on width and height |
| `resize(w, h)` | scales input to exact width and height |
| `reverse()` | reverses a clip, alias to `^-1` |
| `dissolve(sec)` | inserts a dissolve lasting `sec` seconds between neighboring clips in a `>>` timeline |

`text` positions may be written as bare names. The supported descriptive names are `TOP_LEFT`, `TOP`, `TOP_RIGHT`, `LEFT`, `CENTER`, `RIGHT`, `BOTTOM_LEFT`, `BOTTOM`, and `BOTTOM_RIGHT`; two-letter forms such as `TL`, `TC`, and `BR` are also accepted.

A dissolve must appear between two clips, for example `a >> dissolve(0.5) >> b`. Its frames are sampled at intervals of at most 100ms, with the final interval shortened so the transition has the exact requested duration.
