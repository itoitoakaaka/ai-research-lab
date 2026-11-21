"""Streamlit interface for Lab A. Run: streamlit run app.py"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from torch import nn

st.set_page_config(page_title="AI Research Lab A", layout="wide")
st.title("Lab A · Neural-network regression")
st.caption("Interactive synthetic regression experiment with PyTorch")
n = st.sidebar.slider("Training samples", 10, 300, 60, step=10)
noise = st.sidebar.slider("Observation noise (SD)", 0.0, 1.0, 0.2, 0.05)
depth = st.sidebar.slider("Hidden layers", 1, 5, 2)
width = st.sidebar.select_slider("Units per layer", options=[8, 16, 32, 64, 128], value=32)
learning_rate = st.sidebar.select_slider("Learning rate", options=[0.0001, 0.001, 0.01, 0.05], value=0.01)
epochs = st.sidebar.slider("Epochs", 50, 1000, 200, step=50)
seed = st.sidebar.number_input("Random seed", 0, 10000, 0)

if st.button("Train model", type="primary"):
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    fn = lambda x: np.sin(3*x) + 0.3*x
    x = rng.uniform(-2,2,(n,1)).astype("float32")
    y = (fn(x) + rng.normal(0,noise,x.shape)).astype("float32")
    xv = rng.uniform(-2,2,(200,1)).astype("float32")
    yv = (fn(xv) + rng.normal(0,noise,xv.shape)).astype("float32")
    layers, input_width = [], 1
    for _ in range(depth):
        layers.extend([nn.Linear(input_width,width),nn.Tanh()])
        input_width = width
    layers.append(nn.Linear(input_width,1))
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    tx,ty = torch.from_numpy(x),torch.from_numpy(y)
    vx,vy = torch.from_numpy(xv),torch.from_numpy(yv)
    train,valid = [],[]
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(tx),ty)
        loss.backward()
        opt.step()
        with torch.no_grad():
            train.append(float(nn.functional.mse_loss(model(tx),ty)))
            valid.append(float(nn.functional.mse_loss(model(vx),vy)))
    grid = np.linspace(-2,2,400,dtype="float32").reshape(-1,1)
    with torch.no_grad():
        pred = model(torch.from_numpy(grid)).numpy()
    col1,col2 = st.columns(2)
    with col1:
        fig,ax=plt.subplots()
        ax.plot(train,label="Train")
        ax.plot(valid,label="Validation")
        ax.set(xlabel="Epoch",ylabel="MSE",title="Learning curves")
        ax.legend(); st.pyplot(fig)
    with col2:
        fig,ax=plt.subplots()
        ax.scatter(x[:,0],y[:,0],s=12,alpha=.5,label="Training samples")
        ax.plot(grid[:,0],fn(grid)[:,0],label="Ground truth")
        ax.plot(grid[:,0],pred[:,0],label="Model prediction")
        ax.set(xlabel="x",ylabel="y",title="Prediction")
        ax.legend(); st.pyplot(fig)
    st.metric("Final train MSE",f"{train[-1]:.4f}")
    st.metric("Final validation MSE",f"{valid[-1]:.4f}")
    import pandas as pd
    data = pd.DataFrame({"epoch":np.arange(1,epochs+1),"train_mse":train,"val_mse":valid})
    st.download_button("Download training history CSV",data.to_csv(index=False),"training_history.csv","text/csv")
else:
    st.info("Choose experimental parameters and click Train model.")
