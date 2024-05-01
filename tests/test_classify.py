from typing import List, Union

import pytest
from pydantic import BaseModel, ValidationError

from yogahub.models import YogaClassifier


class PredictionResponse(BaseModel):
    Target: str
    PotentialCandidate: Union[List[str], str]
    Gesture: str


def test_yoga_classifier_predict():
    # Arrange
    model = YogaClassifier(pretrained=None)
    test_image_path = "example/test.png"

    # Act
    output = model.predict(images=test_image_path, convert_to_chinese=True)

    # Assert
    try:
        # Attempt to create a PredictionResponse object with the output
        response = PredictionResponse(**output)

        # Additional checks can be added here if necessary
        assert response.Target is not None
        assert response.Gesture is not None
        assert response.PotentialCandidate is not None
    except ValidationError as e:
        pytest.fail(f"Output validation failed: {e}")
