# puzzlomino
Estimate puzzle completion with image recognition

# Example Usage
![puzzle.jpg](/documentation/puzzle.jpg)

```py
import cv2
from puzzlomino import puzzlomino

image = cv2.imread("./documentation/puzzle.jpg")
preprocessed = puzzlomino.preprocess(image)
puzzle_contour = puzzlomino.get_puzzle_contour(preprocessed)

plt.imshow(puzzle_contour.overlay_on(image))
print(f"{puzzle_contour.area():0.1%} complete")
```

![puzzle-contour.png](/documentation/puzzle-contour.png)

```
16.5% complete
```
