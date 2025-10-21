# Heuristic Hand Gestures

This template enables the heuristic gesture recognizer and overlays a simple
dashboard that lists the gesture state for each hand. It is useful for quickly
validating pinch, open-palm, fist, thumbs-up, point, and spread detections on
devices like Quest.

The dashboard subscribes to the shared `GestureRecognition` subsystem exposed at
`xb.core.gestureRecognition`, highlighting gestures as they emit
`gesturestart`, `gestureupdate`, and `gestureend` events.
