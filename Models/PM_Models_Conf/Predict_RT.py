import numpy as np
import pandas as pd
import joblib

# 1. Load saved preprocessor and model (update paths if needed)
preproc = joblib.load('../PM_Models_Conf/Models/JNN_RT_preproc_binary_rt95.pkl')
model = joblib.load('../PM_Models_Conf/Models/JNN_RT_XGB_binary_rt95.pkl')


def predict_response_time_point(

        actual_rpm: float,
        adv1: int, adv2: int,
        analytics1: int, analytics2: int,
        breaking: int,
        content1: int, content2: int,
        #gateway: int,
        media1: int, media2: int,
        recommendation1: int, recommendation2: int,
) -> float:
    """
    Predicts average response time (ms) for a given JNN configuration/workload point.

    Notes:
    - The model actually uses the binary service flags + num_users + actual_rpm.
      The 'config' column is included only for compatibility with your older style.
    """

    # Assemble input row (columns must match training features)
    row = pd.DataFrame([{

        'actual_rpm': actual_rpm,

        'adv1': adv1,
        'adv2': adv2,
        'analytics1': analytics1,
        'analytics2': analytics2,
        'breaking': breaking,
        'content1': content1,
        'content2': content2,
        #'gateway': gateway,
        'media1': media1,
        'media2': media2,
        'recommendation1': recommendation1,
        'recommendation2': recommendation2,

    }])

    # 2. Transform features
    X_t = preproc.transform(row)

    # 3. Predict log-RT and back-transform to milliseconds
    log_pred = model.predict(X_t)
    rt_pred_ms = np.expm1(log_pred)  # inverse of log1p used in training

    return float(rt_pred_ms)


if __name__ == "__main__":
    # Example 1: light config (similar to config with content1 + gateway)
    # example_rt_light = predict_response_time_point(
    #     config=1,
    #     num_users=20,
    #     actual_rpm=10,
    #     adv1=0, adv2=0,
    #     analytics1=0, analytics2=1,
    #     breaking=1,
    #     content1=0, content2=1,
    #    # gateway=1,
    #     media1=0, media2=1,
    #     recommendation1=0, recommendation2=1,
    # )
    # print(f"[Light config] Predicted RT: {example_rt_light:.2f} ms")
    #82,"content2, media1, analytics2, recommendation2, adv1",1,0,0,1,0,0,1,1,0,0,1,1.12,0.8253968253968256

    example_rt_heavy = predict_response_time_point(
        actual_rpm=960.0,
        adv1=1, adv2=0,
        analytics1=0, analytics2=1,
        breaking=0,
        content1=0, content2=1,
       # gateway=1,
        media1=1, media2=0,
        recommendation1=0, recommendation2=1,

    )
    print(f"[Heavy config] Predicted RT: {example_rt_heavy:.2f} ms")
