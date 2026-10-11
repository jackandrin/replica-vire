import { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import TradingDashboard from './trading/TradingDashboard'
import './index.css'

// Two pages, no router dependency: /dashboard is the paper-trading dashboard, everything else is the research site.
const isDashboard = /^\/dashboard\/?$/.test(window.location.pathname)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isDashboard ? <TradingDashboard /> : <App />}
  </StrictMode>,
)
