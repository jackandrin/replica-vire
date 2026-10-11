import { useState, type ReactNode } from "react"

export type Screen = "onboarding" | "results" | "performance"
export type Profile = {
  budget: number
  risk: string
  horizon: string
  sectors: string[]
}
export type IconName = "arrow" | "briefcase" | "chart" | "check" | "chevron" | "clock" | "close" | "edit" | "growth" | "leaf" | "menu" | "shield" | "sparkle" | "user"

const iconPaths: Record<IconName, ReactNode> = {
  arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  briefcase: (
    <>
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12h18" />
    </>
  ),
  chart: (
    <>
      <path d="M4 19V9m6 10V5m6 14v-7m5 7H2" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m8 10 4 4 4-4" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  close: <path d="m6 6 12 12M18 6 6 18" />,
  edit: (
    <>
      <path d="m14 5 5 5M4 20l3.5-.8L19 7.7a2 2 0 0 0-2.7-2.7L4.8 16.5 4 20Z" />
    </>
  ),
  growth: (
    <>
      <path d="m4 17 5-5 4 3 7-8" />
      <path d="M15 7h5v5" />
    </>
  ),
  leaf: (
    <>
      <path d="M20 4C11 4 5 8 5 14c0 3 2 5 5 5 6 0 10-6 10-15Z" />
      <path d="M4 20c3-5 7-8 12-11" />
    </>
  ),
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  shield: (
    <>
      <path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6l-7-3Z" />
      <path d="m9 12 2 2 4-4" />
    </>
  ),
  sparkle: (
    <>
      <path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z" />
      <path d="m18.5 15 .7 2.2 2.3.8-2.3.8-.7 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="3" />
      <path d="M5 20c.7-4 3-6 7-6s6.3 2 7 6" />
    </>
  ),
}

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {iconPaths[name]}
    </svg>
  )
}

export function Button({
  children,
  kind = "primary",
  className = "",
  onClick,
  type = "button",
}: {
  children: ReactNode
  kind?: "primary" | "secondary" | "ghost"
  className?: string
  onClick?: () => void
  type?: "button" | "submit"
}) {
  return (
    <button
      type={type}
      className={`button button-${kind} ${className}`}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode
  tone?: "neutral" | "green" | "blue" | "orange"
}) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

function Logo() {
  return (
    <div className="logo" aria-label="ReplicaVire">
      <span className="logo-mark">
        <i />
        <i />
        <i />
      </span>
      <span className="wordmark">ReplicaVire</span>
      <span className="descriptor">EQUITY RESEARCH</span>
    </div>
  )
}

export function Shell({
  screen,
  setScreen,
  onExit,
  children,
}: {
  screen: Screen
  setScreen: (screen: Screen) => void
  onExit: () => void
  children: ReactNode
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  return (
    <div className="app-shell">
      <header className="topbar">
        <Logo />
        <nav
          className={`nav-tabs ${menuOpen ? "nav-open" : ""}`}
          aria-label="Main navigation"
        >
          <button
            className={
              screen === "results" || screen === "onboarding" ? "active" : ""
            }
            onClick={() => {
              setScreen("results")
              setMenuOpen(false)
            }}
          >
            Portfolio
          </button>
          <button
            className={screen === "performance" ? "active" : ""}
            onClick={() => {
              setScreen("performance")
              setMenuOpen(false)
            }}
          >
            Performance
          </button>
          <button
            onClick={() => {
              setMenuOpen(false)
              onExit()
            }}
          >
            Research
          </button>
        </nav>
        <div className="nav-actions">
          <button
            className="mobile-menu"
            aria-label="Toggle navigation"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <Icon name={menuOpen ? "close" : "menu"} />
          </button>
        </div>
      </header>
      <main>{children}</main>
      <footer className="disclaimer">
        <Icon name="shield" size={15} />
        <span>
          <strong>Simulated trading.</strong> Not financial advice. No real
          money is involved.
        </span>
      </footer>
    </div>
  )
}

const budgets = [50, 100, 250, 500]
export const sectors = [
  "Technology",
  "Communication",
  "Consumer",
  "Financial",
  "Healthcare",
  "Energy",
  "No preference",
]
const risks = [
  {
    name: "Conservative",
    desc: "Prioritize stability and smoother returns.",
    icon: "shield" as IconName,
  },
  {
    name: "Balanced",
    desc: "A steady mix of growth and resilience.",
    icon: "leaf" as IconName,
  },
  {
    name: "Growth",
    desc: "More upside potential, with bigger swings.",
    icon: "growth" as IconName,
  },
]

function Progress({ step }: { step: number }) {
  return (
    <div className="progress-wrap">
      <div className="progress-copy">
        <span>Portfolio setup</span>
        <strong>{step} of 4</strong>
      </div>
      <div className="progress-track">
        <span style={{ width: `${step * 25}%` }} />
      </div>
      <div className="progress-labels">
        <span className={step >= 1 ? "done" : ""}>Budget</span>
        <span className={step >= 2 ? "done" : ""}>Risk</span>
        <span className={step >= 3 ? "done" : ""}>Timeline</span>
        <span className={step >= 4 ? "done" : ""}>Preferences</span>
      </div>
    </div>
  )
}

export function Onboarding({
  initial,
  onComplete,
}: {
  initial: Profile
  onComplete: (profile: Profile) => void
}) {
  const [step, setStep] = useState(1)
  const [budgetText, setBudgetText] = useState(String(initial.budget))
  const budget = Number(budgetText)
  const budgetValid = Number.isFinite(budget) && budget >= 10
  const [risk, setRisk] = useState(initial.risk)
  const [horizon, setHorizon] = useState(initial.horizon)
  const [selectedSectors, setSelectedSectors] = useState<string[]>(
    initial.sectors,
  )

  const toggleSector = (sector: string) => {
    if (sector === "No preference") return setSelectedSectors(["No preference"])
    setSelectedSectors((current) =>
      current.includes(sector)
        ? current.filter((item) => item !== sector)
        : [...current.filter((item) => item !== "No preference"), sector],
    )
  }

  return (
    <div className="onboarding-page">
      <div className="onboarding-aside">
        <div className="aside-content">
          <Badge tone="green">
            <Icon name="sparkle" size={14} /> Built around you
          </Badge>
          <h1>Investing, without the guesswork.</h1>
          <p>
            Tell us what matters to you. We’ll build a simple, diversified
            portfolio and explain every choice.
          </p>
          <div className="trust-list">
            <div>
              <span>
                <Icon name="check" size={14} />
              </span>
              <p>
                <strong>Plain-English picks</strong>
                <small>No jargon, just clear reasoning.</small>
              </p>
            </div>
            <div>
              <span>
                <Icon name="check" size={14} />
              </span>
              <p>
                <strong>Made for small starts</strong>
                <small>Build a plan from just $50.</small>
              </p>
            </div>
            <div>
              <span>
                <Icon name="check" size={14} />
              </span>
              <p>
                <strong>Zero-stakes practice</strong>
                <small>Learn with simulated money.</small>
              </p>
            </div>
          </div>
        </div>
        <p className="aside-quote">“A smarter first step into investing.”</p>
      </div>

      <section className="onboarding-main">
        <div className="onboarding-card">
          <Progress step={step} />
          <div className="step-content" key={step}>
            {step === 1 && (
              <>
                <span className="eyebrow">Step one</span>
                <h2>What are you starting with?</h2>
                <p className="step-description">
                  Choose a practice budget. You can always change this later.
                </p>
                <label className="money-input">
                  <span>$</span>
                  <input
                    value={budgetText}
                    onChange={(e) =>
                      setBudgetText(e.target.value.replace(/[^0-9.]/g, ""))
                    }
                    inputMode="decimal"
                    aria-label="Budget amount"
                  />
                </label>
                {!budgetValid && (
                  <p className="step-description">Enter at least $10.</p>
                )}
                <div className="chip-grid">
                  {budgets.map((item) => (
                    <button
                      key={item}
                      className={`chip ${budget === item ? "selected" : ""}`}
                      onClick={() => setBudgetText(String(item))}
                    >
                      {item === budget && <Icon name="check" size={14} />}
                      ${item}
                    </button>
                  ))}
                </div>
              </>
            )}
            {step === 2 && (
              <>
                <span className="eyebrow">Step two</span>
                <h2>How do you feel about risk?</h2>
                <p className="step-description">
                  There’s no wrong answer. Pick what feels most comfortable.
                </p>
                <div className="risk-list">
                  {risks.map((item) => (
                    <button
                      key={item.name}
                      className={`risk-card ${
                        risk === item.name ? "selected" : ""
                      }`}
                      onClick={() => setRisk(item.name)}
                    >
                      <span className="risk-icon">
                        <Icon name={item.icon} />
                      </span>
                      <span>
                        <strong>{item.name}</strong>
                        <small>{item.desc}</small>
                      </span>
                      <span className="radio-dot">
                        {risk === item.name && <span />}
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}
            {step === 3 && (
              <>
                <span className="eyebrow">Step three</span>
                <h2>When might you need this money?</h2>
                <p className="step-description">
                  Longer timelines can usually handle more market movement.
                </p>
                <div className="horizon-visual">
                  <div className="horizon-orbit">
                    <Icon name="clock" size={32} />
                  </div>
                  <strong>{horizon}</strong>
                  <span>Your investing horizon</span>
                </div>
                <div className="segmented">
                  {["1 year", "3 years", "5+ years", "10+ years"].map(
                    (item) => (
                      <button
                        key={item}
                        className={horizon === item ? "selected" : ""}
                        onClick={() => setHorizon(item)}
                      >
                        {item}
                      </button>
                    ),
                  )}
                </div>
              </>
            )}
            {step === 4 && (
              <>
                <span className="eyebrow">Last step</span>
                <h2>Anything you’re excited about?</h2>
                <p className="step-description">
                  Optional. Pick as many sectors as you like, or leave it to us.
                </p>
                <div className="sector-grid">
                  {sectors.map((item) => (
                    <button
                      key={item}
                      className={`sector-chip ${
                        selectedSectors.includes(item) ? "selected" : ""
                      }`}
                      onClick={() => toggleSector(item)}
                    >
                      {selectedSectors.includes(item) && (
                        <Icon name="check" size={15} />
                      )}
                      {item}
                    </button>
                  ))}
                </div>
                <div className="mini-summary">
                  <span>
                    <Icon name="sparkle" size={16} />
                  </span>
                  <p>
                    <strong>Your starting plan</strong>
                    <small>
                      ${budgetValid ? budget : "—"} · {risk} · {horizon}
                    </small>
                  </p>
                </div>
              </>
            )}
          </div>
          <div className="step-actions">
            {step > 1 && (
              <Button kind="ghost" onClick={() => setStep(step - 1)}>
                Back
              </Button>
            )}
            <Button
              className="next-button"
              onClick={() => {
                if (!budgetValid) return setStep(1)
                if (step < 4) return setStep(step + 1)
                onComplete({
                  budget,
                  risk,
                  horizon,
                  sectors: selectedSectors,
                })
              }}
            >
              {step === 4 ? "Build my portfolio" : "Continue"}{" "}
              <Icon name="arrow" />
            </Button>
          </div>
          <p className="paper-note">
            <Icon name="shield" size={14} /> Paper trading only. Not financial
            advice.
          </p>
        </div>
      </section>
    </div>
  )
}

