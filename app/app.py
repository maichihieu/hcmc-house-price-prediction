"""
HCMC House Price Prediction - Desktop Application
==================================================
A CustomTkinter GUI application that predicts Ho Chi Minh City
house prices using an ensemble of CatBoost and XGBoost models.

Usage:
    python app.py

Requirements:
    pip install customtkinter pandas numpy joblib scikit-learn catboost xgboost
"""

import sys
import os
import json
import time
import traceback

import customtkinter as ctk
import tkinter.messagebox as msg_box
import pandas as pd
import numpy as np
import joblib

try:
    from ctk_scrollable_dropdown import CTkScrollableDropdown
except ImportError:
    msg_box.showerror(
        "Missing File",
        "Cannot find 'ctk_scrollable_dropdown.py'.\n"
        "Please make sure it is in the same directory as app.py."
    )
    sys.exit(1)

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Custom Transformers & Wrappers (required for unpickling trained pipelines)
# ---------------------------------------------------------------------------

def log_transform(X: np.ndarray) -> np.ndarray:
    """Apply log1p transformation to feature array."""
    return np.log1p(X)


class RareLabelGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categorical labels (below a frequency threshold)
    into a single 'Other' category.
    """

    def __init__(self, threshold: int = 5, replacement: str = "Khác"):
        self.threshold = threshold
        self.replacement = replacement
        self.valid_labels_ = None

    def fit(self, X, y=None):
        series = self._to_series(X)
        counts = series.value_counts()
        self.valid_labels_ = counts[counts > self.threshold].index
        return self

    def transform(self, X):
        series = self._to_series(X)
        return (
            series
            .where(series.isin(self.valid_labels_), self.replacement)
            .values
            .reshape(-1, 1)
        )

    @staticmethod
    def _to_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, np.ndarray):
            return pd.Series(X.ravel())
        return X


class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoder for categorical features.
    Uses Bayesian smoothing to blend category mean with global mean.
    """

    def __init__(self, smoothing: int = 10):
        self.smoothing = smoothing
        self.encoding_map_ = None
        self.global_mean_ = None

    def fit(self, X, y):
        x_series = pd.Series(X.ravel())
        y_series = pd.Series(y)
        stats = pd.DataFrame({"group": x_series, "target": y_series})
        agg = stats.groupby("group")["target"].agg(["count", "mean"])
        self.global_mean_ = y_series.mean()
        smoothing_weight = agg["count"] / (agg["count"] + self.smoothing)
        self.encoding_map_ = (
            smoothing_weight * agg["mean"]
            + (1 - smoothing_weight) * self.global_mean_
        )
        return self

    def transform(self, X):
        x_series = pd.Series(X.ravel())
        return (
            x_series
            .map(self.encoding_map_)
            .fillna(self.global_mean_)
            .values
            .reshape(-1, 1)
        )


class CatBoostWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for CatBoostRegressor."""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        random_state: int = 3,
        cat_features=None,
        verbose: int = 0,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.cat_features = cat_features
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            cat_features=self.cat_features,
            verbose=self.verbose,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Input fields: (display label, internal DataFrame key)
INPUT_FIELDS = [
    ("Area (m²)",          "dien_tich"),
    ("Bedrooms",            "so_pn"),
    ("Floors",              "so_tang"),
    ("Road Width (m)",      "duong_vao"),
    ("Frontage (m)",        "mat_tien"),
    ("Ward / District",     "phuong_xa"),
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__))


class HousePricePredictorApp(ctk.CTk):
    """Main application window for HCMC House Price Prediction."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.title("AI Real Estate Valuator – HCMC House Price Prediction")
        self.geometry("1100x650")
        self.resizable(False, False)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.catboost_model = None
        self.xgboost_model  = None
        self.ward_list: list[str] = ["Loading..."]

        self._register_custom_classes()
        self._load_resources()
        self._build_sidebar()
        self._build_main_panel()

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------

    @staticmethod
    def _get_resource_path(relative_path: str) -> str:
        """Return absolute path; supports PyInstaller bundles."""
        try:
            base = sys._MEIPASS  # type: ignore[attr-defined]
        except AttributeError:
            base = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base, relative_path)

    def _register_custom_classes(self):
        """
        Inject custom classes into __main__ so joblib can unpickle
        pipelines that were saved from a training script.
        """
        import __main__
        __main__.log_transform_feature = log_transform  # backward-compat alias
        __main__.log_transform         = log_transform
        __main__.RareLabelGrouper      = RareLabelGrouper
        __main__.CustomTargetEncoder   = CustomTargetEncoder
        __main__.CatBoostWrapper       = CatBoostWrapper

    def _load_resources(self):
        """Load ML models and ward list from disk."""
        try:
            catboost_path = self._get_resource_path(
                os.path.join("..", "models", "catboost_pipeline.pkl")
            )
            xgboost_path = self._get_resource_path(
                os.path.join("..", "models", "xgboost_pipeline.pkl")
            )
            wards_path = self._get_resource_path("districts_wards.json")

            if os.path.exists(catboost_path):
                self.catboost_model = joblib.load(catboost_path)

            if os.path.exists(xgboost_path):
                self.xgboost_model = joblib.load(xgboost_path)

            if os.path.exists(wards_path):
                with open(wards_path, "r", encoding="utf-8") as fh:
                    self.ward_list = json.load(fh)

        except Exception as exc:
            msg_box.showerror("Resource Load Error", str(exc))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_sidebar(self):
        """Build the left sidebar with input fields."""
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar,
            text="INPUT PARAMETERS",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(padx=20, pady=(30, 20))

        input_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        input_container.pack(fill="both", expand=True)

        self.input_widgets: dict[str, ctk.CTkEntry] = {}

        for label_text, field_key in INPUT_FIELDS:
            ctk.CTkLabel(
                input_container,
                text=label_text,
                anchor="w",
                font=ctk.CTkFont(size=13),
            ).pack(fill="x", padx=20, pady=(10, 0))

            if field_key == "phuong_xa":
                entry = ctk.CTkEntry(
                    input_container,
                    placeholder_text="Search or select ward..."
                )
                CTkScrollableDropdown(
                    attach=entry,
                    values=self.ward_list,
                    command=lambda val: None,
                    autocomplete=True,
                    height=150,
                )
            else:
                entry = ctk.CTkEntry(input_container, placeholder_text="0")

            entry.pack(fill="x", padx=20, pady=(5, 5))
            self.input_widgets[field_key] = entry

        self.predict_button = ctk.CTkButton(
            self.sidebar,
            text="PREDICT PRICE",
            command=self._on_predict_clicked,
            height=45,
            fg_color="#e67e22",
            hover_color="#d35400",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.predict_button.pack(padx=20, pady=30, fill="x", side="bottom")

    def _build_main_panel(self):
        """Build the right panel showing prediction results."""
        main_panel = ctk.CTkFrame(self, fg_color="transparent")
        main_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        ctk.CTkLabel(
            main_panel,
            text="HCMC REAL ESTATE VALUATION SYSTEM",
            font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
        ).pack(pady=(40, 50))

        result_card = ctk.CTkFrame(
            main_panel, fg_color=("#dfe6e9", "#2d3436"), corner_radius=20
        )
        result_card.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            result_card,
            text="Estimated Price",
            font=ctk.CTkFont(size=16),
        ).pack(pady=(25, 5))

        self.price_label = ctk.CTkLabel(
            result_card,
            text="--- VND",
            font=ctk.CTkFont(family="Arial", size=42, weight="bold"),
            text_color="#2ecc71",
        )
        self.price_label.pack(pady=(0, 25))

        self.detail_label = ctk.CTkLabel(
            result_card,
            text="Enter parameters and click Predict",
            text_color="#b2bec3",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.detail_label.pack(pady=(0, 20))

        self.progress_bar = ctk.CTkProgressBar(
            main_panel, width=400, progress_color="#2ecc71"
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=40)

    # ------------------------------------------------------------------
    # Prediction logic
    # ------------------------------------------------------------------

    def _on_predict_clicked(self):
        """Handle predict button click: validate, predict, display result."""
        self._set_loading_state(True)

        # Animate progress bar
        for step in range(101):
            self.progress_bar.set(step / 100)
            self.update()
            time.sleep(0.02)

        try:
            user_input = self._collect_and_validate_inputs()
            if user_input is None:
                return

            prediction_df = self._build_prediction_dataframe(user_input)
            price_vnd, detail_text = self._run_ensemble_prediction(prediction_df)

            if price_vnd is None:
                return

            formatted_price = self._format_price(price_vnd)
            self.price_label.configure(
                text=f"{formatted_price} VND", text_color="#2ecc71"
            )
            self.detail_label.configure(text=detail_text)

        except ValueError:
            msg_box.showerror("Input Error", "Please enter valid numeric values!")
        except Exception as exc:
            msg_box.showerror("Prediction Error", f"Details:\n{exc}")
            traceback.print_exc()
        finally:
            self._set_loading_state(False)

    def _collect_and_validate_inputs(self) -> dict | None:
        """Read and validate all entry widgets. Returns dict or None on error."""
        data: dict[str, str] = {}
        for key, widget in self.input_widgets.items():
            value = widget.get().strip()
            if not value or value == "Loading...":
                msg_box.showwarning(
                    "Missing Input", f"Please fill in the field: {key}"
                )
                return None
            data[key] = value

        if data["phuong_xa"] not in self.ward_list:
            if not msg_box.askyesno(
                "Warning",
                f"Ward '{data['phuong_xa']}' is unfamiliar.\n"
                "Results may be inaccurate. Continue?",
            ):
                return None

        return data

    @staticmethod
    def _build_prediction_dataframe(user_input: dict) -> pd.DataFrame:
        """Convert validated user input dict into a model-ready DataFrame."""
        return pd.DataFrame(
            {
                "dien_tich": [float(user_input["dien_tich"])],
                "phuong_xa":  [str(user_input["phuong_xa"])],
                "duong_vao":  [float(user_input["duong_vao"])],
                "mat_tien":   [float(user_input["mat_tien"])],
                "so_tang":    [float(user_input["so_tang"])],
                "so_pn":      [float(user_input["so_pn"])],
            }
        )

    def _run_ensemble_prediction(
        self, df: pd.DataFrame
    ) -> tuple[float | None, str]:
        """
        Run CatBoost and XGBoost models, then return the ensemble average
        and a human-readable detail string.
        """
        log_pred_cat = (
            self.catboost_model.predict(df)[0] if self.catboost_model else None
        )
        log_pred_xgb = (
            self.xgboost_model.predict(df)[0] if self.xgboost_model else None
        )

        if log_pred_cat is not None and log_pred_xgb is not None:
            price_cat = np.expm1(log_pred_cat)
            price_xgb = np.expm1(log_pred_xgb)
            ensemble_price = (price_cat + price_xgb) / 2
            detail = (
                f"CatBoost: {price_cat / 1e9:.2f} B  |  "
                f"XGBoost: {price_xgb / 1e9:.2f} B"
            )
            return ensemble_price, detail

        if log_pred_cat is not None:
            return np.expm1(log_pred_cat), "Result from CatBoost only"

        if log_pred_xgb is not None:
            return np.expm1(log_pred_xgb), "Result from XGBoost only"

        msg_box.showerror("Error", "No model found!")
        return None, ""

    @staticmethod
    def _format_price(price_vnd: float) -> str:
        """Format a VND price value into a human-readable string."""
        if price_vnd >= 1e9:
            return f"{price_vnd / 1e9:,.2f} Billion"
        return f"{price_vnd / 1e6:,.0f} Million"

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_loading_state(self, is_loading: bool):
        """Toggle loading state: disable button and show loading text."""
        if is_loading:
            self.price_label.configure(text="Calculating...", text_color="orange")
            self.detail_label.configure(text="Please wait...")
            self.progress_bar.set(0)
            self.predict_button.configure(state="disabled")
        else:
            self.predict_button.configure(state="normal")
            self.progress_bar.set(1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = HousePricePredictorApp()
    app.mainloop()
