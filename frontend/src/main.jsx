import React,{useEffect,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {BarChart,Bar,LineChart,Line,XAxis,YAxis,Tooltip,CartesianGrid,ResponsiveContainer} from 'recharts';
import './styles.css';

const api=(path,opts)=>fetch(path,opts).then(r=>{if(!r.ok)throw new Error('API request failed');return r.json()});
const initial={year:2018,present_price:8.5,kms_driven:45000,fuel_type:'Petrol',seller_type:'Dealer',transmission:'Manual',owner:0};
function App(){
 const [form,setForm]=useState(initial); const [prediction,setPrediction]=useState(null); const [eda,setEda]=useState(null); const [model,setModel]=useState(null); const [error,setError]=useState('');
 useEffect(()=>{Promise.all([api('/api/eda'),api('/api/model-info')]).then(([a,m])=>{setEda(a);setModel(m)}).catch(e=>setError('Backend not connected. Run the backend or deploy the Vercel services.'))},[]);
 const set=(k,v)=>setForm({...form,[k]:v});
 const predict=async e=>{e.preventDefault();setError('');try{setPrediction(await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...form,year:Number(form.year),present_price:Number(form.present_price),kms_driven:Number(form.kms_driven),owner:Number(form.owner)})}))}catch(e){setError('Prediction failed. Please check the backend.')}};
 return <div className="app">
  <header><div className="brand"><span className="logo">₹</span><div><h1>Used Car Price Estimator</h1><p>Team 12 • Regression + Random Forest</p></div></div><div className="team">DARSHAN H N • LIKHITH M • NIKESH</div></header>
  <main>
   <section className="hero"><div><span className="pill">MACHINE LEARNING CASE STUDY</span><h2>Estimate a fair resale price in seconds.</h2><p>Enter vehicle details and our Random Forest Regression model estimates the used-car selling price in Indian Lakhs.</p></div><div className="hero-stat"><b>301</b><span>Dataset records</span><b>9</b><span>Original features</span></div></section>
   {error&&<div className="error">{error}</div>}
   <section className="grid">
    <div className="card form-card"><div className="card-head"><h3>Vehicle Details</h3><span>Step 1</span></div><form onSubmit={predict}>
      <label>Manufacturing Year<input type="number" min="1990" max="2030" value={form.year} onChange={e=>set('year',e.target.value)}/></label>
      <label>Present Price (₹ Lakh)<input type="number" step="0.01" min="0.1" value={form.present_price} onChange={e=>set('present_price',e.target.value)}/></label>
      <label>Kilometers Driven<input type="number" min="0" value={form.kms_driven} onChange={e=>set('kms_driven',e.target.value)}/></label>
      <div className="two"><label>Fuel Type<select value={form.fuel_type} onChange={e=>set('fuel_type',e.target.value)}><option>Petrol</option><option>Diesel</option><option>CNG</option></select></label><label>Seller Type<select value={form.seller_type} onChange={e=>set('seller_type',e.target.value)}><option>Dealer</option><option>Individual</option></select></label></div>
      <div className="two"><label>Transmission<select value={form.transmission} onChange={e=>set('transmission',e.target.value)}><option>Manual</option><option>Automatic</option></select></label><label>Previous Owners<input type="number" min="0" max="5" value={form.owner} onChange={e=>set('owner',e.target.value)}/></label></div>
      <button>Predict Used-Car Price →</button>
    </form></div>
    <div className="card result-card"><div className="card-head"><h3>Prediction</h3><span>Step 2</span></div>{prediction?<><div className="result-label">Recommended estimate</div><div className="price">₹ {prediction.random_forest_price_lakh.toFixed(2)} <small>Lakh</small></div><div className="compare"><div><span>Random Forest</span><b>₹ {prediction.random_forest_price_lakh.toFixed(2)} L</b></div><div><span>Linear Regression</span><b>₹ {prediction.linear_regression_price_lakh.toFixed(2)} L</b></div></div><div className="winner">🏆 Recommended model: Random Forest Regressor</div></>:<div className="empty"><div>🚗</div><h4>Your estimate will appear here</h4><p>Fill in the vehicle details and click predict.</p></div>}</div>
   </section>
   <section className="card"><div className="card-head"><div><h3>EDA Dashboard</h3><p>Data distribution and relationships used in the case study.</p></div><span>Step 3</span></div>{eda&&<div className="charts"><div className="chart"><h4>Average Selling Price by Fuel</h4><ResponsiveContainer width="100%" height={250}><BarChart data={Object.entries(eda.fuel_avg).map(([name,value])=>({name,value}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name"/><YAxis/><Tooltip/><Bar dataKey="value"/></BarChart></ResponsiveContainer></div><div className="chart"><h4>Average Price by Transmission</h4><ResponsiveContainer width="100%" height={250}><BarChart data={Object.entries(eda.transmission_avg).map(([name,value])=>({name,value}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name"/><YAxis/><Tooltip/><Bar dataKey="value"/></BarChart></ResponsiveContainer></div><div className="chart wide"><h4>Year vs Average Selling Price</h4><ResponsiveContainer width="100%" height={270}><LineChart data={eda.year_trend}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="Year"/><YAxis/><Tooltip/><Line type="monotone" dataKey="avg_price" strokeWidth={3}/></LineChart></ResponsiveContainer></div></div>}</section>
   <section className="grid three"><div className="mini"><b>Raw rows</b><strong>{eda?.rows_raw??'—'}</strong><span>Before cleaning</span></div><div className="mini"><b>Clean rows</b><strong>{eda?.rows_clean??'—'}</strong><span>After duplicate removal</span></div><div className="mini"><b>Duplicates removed</b><strong>{eda?.duplicates_removed??'—'}</strong><span>Preprocessing check</span></div></section>
   <section className="card"><div className="card-head"><div><h3>Model Performance</h3><p>Regression metrics from the held-out test set.</p></div></div>{model&&<div className="metrics"><div className="metric"><span>Linear Regression</span><b>R² {model.metrics['Linear Regression'].r2}</b><small>MAE {model.metrics['Linear Regression'].mae} L • RMSE {model.metrics['Linear Regression'].rmse} L</small></div><div className="metric best"><span>Random Forest</span><b>R² {model.metrics['Random Forest'].r2}</b><small>MAE {model.metrics['Random Forest'].mae} L • RMSE {model.metrics['Random Forest'].rmse} L</small></div></div>}</section>
   <section className="method"><div><h3>Project Pipeline</h3><p>Dataset → preprocessing → EDA → train/test split → regression models → evaluation → prediction.</p></div><div className="steps"><span>01 Data</span><span>02 Clean</span><span>03 EDA</span><span>04 Train</span><span>05 Predict</span></div></section>
  </main><footer><b>Team 12</b> — DARSHAN H N (1VA23CD025) • LIKHITH M (1VA23CD044) • NIKESH (1VA23CD058)<br/><span>Used Car Price Estimation using Regression / Random Forest</span></footer>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
