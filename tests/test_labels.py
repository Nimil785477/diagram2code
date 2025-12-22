from diagram2code.labels import to_valid_identifier


def test_to_valid_identifier_style2():
    assert to_valid_identifier("Step_1_Load_Data", "fallback") == "Step_1_Load_Data"
    assert to_valid_identifier("Step 2 - Train Model", "fallback") == "Step_2_Train_Model"
    assert to_valid_identifier("1Bad", "fallback") == "_1Bad"
