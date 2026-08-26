import pandas as pd
import numpy as np

from salary_survey.data import clean_domain_errors, filter_salary_scope
from salary_survey.pipelines import build_preprocessor
from salary_survey.transformers import OtherDatabasesEncoder


def test_domain_cleaning_removes_counter_and_invalid_experience():
    frame = pd.DataFrame(
        {
            "YearsWithThisDatabase": [5, 61],
            "YearsWithThisTypeOfJob": [2, 80],
            "Counter": [1, 1],
        }
    )
    result = clean_domain_errors(frame)
    assert "Counter" not in result
    assert pd.isna(result.loc[1, "YearsWithThisDatabase"])
    assert pd.isna(result.loc[1, "YearsWithThisTypeOfJob"])


def test_other_databases_is_multilabel_encoded():
    train = pd.DataFrame({"OtherDatabases": ["Oracle, PostgreSQL", "Oracle", "MySQL"]})
    encoder = OtherDatabasesEncoder(min_frequency=0, max_features=10).fit(train)
    transformed = encoder.transform(pd.DataFrame({"OtherDatabases": ["Oracle, MySQL"]}))
    assert transformed.shape == (1, 3)
    assert transformed.sum() == 2


def test_domain_cleaning_handles_both_invalid_bounds_without_mutating_input():
    frame = pd.DataFrame(
        {
            "YearsWithThisDatabase": [-1, 0, 60],
            "YearsWithThisTypeOfJob": [61, 3, 8],
        }
    )
    original = frame.copy(deep=True)

    result = clean_domain_errors(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert pd.isna(result.loc[0, "YearsWithThisDatabase"])
    assert pd.isna(result.loc[0, "YearsWithThisTypeOfJob"])
    assert result.loc[1, "Experience_Level"] == "Junior"
    assert result.loc[2, "Experience_Level"] == "Mid"


def test_salary_scope_is_inclusive_and_does_not_modify_input():
    frame = pd.DataFrame({"SalaryUSD": [4_999, 5_000, 500_000, 500_001, np.nan]})
    result = filter_salary_scope(frame)
    assert result["SalaryUSD"].tolist() == [5_000, 500_000]
    assert len(frame) == 5


def test_preprocessor_imputes_none_instead_of_creating_none_category():
    frame = pd.DataFrame(
        {
            "category": pd.Series(["known", None, "known"], dtype="object"),
            "number": [1.0, np.nan, 3.0],
        }
    )
    preprocessor = build_preprocessor(frame)

    transformed = preprocessor.fit_transform(frame)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert transformed.shape[0] == len(frame)
    assert np.isfinite(transformed).all()
    assert not any(name.endswith("_None") for name in feature_names)


def test_preprocessor_ignores_unseen_categories_and_database_tokens():
    train = pd.DataFrame(
        {
            "category": ["a", "b"],
            "number": [1.0, 2.0],
            "OtherDatabases": ["Oracle", "PostgreSQL"],
        }
    )
    test = pd.DataFrame(
        {"category": ["new"], "number": [np.nan], "OtherDatabases": ["UnknownDB"]}
    )
    preprocessor = build_preprocessor(train, other_db_min_frequency=0)
    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
