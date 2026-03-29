class Vector2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  add(v) {
    return new Vector2(this.x + v.x, this.y + v.y);
  }

  sub(v) {
    return new Vector2(this.x - v.x, this.y - v.y);
  }

  scale(s) {
    return new Vector2(this.x * s, this.y * s);
  }

  mag() {
    return Math.hypot(this.x, this.y);
  }

  magSq() {
    return this.x * this.x + this.y * this.y;
  }

  normalize() {
    const m = this.mag();
    return m > 0 ? this.scale(1 / m) : new Vector2(0, 0);
  }

  clone() {
    return new Vector2(this.x, this.y);
  }
}

class Car {
  constructor(length = 200, height = 80, noseAngle = 20, rearAngle = 15) {
    this.length = length;
    this.height = height;
    this.noseAngle = noseAngle;
    this.rearAngle = rearAngle;
    this.vertices = [];
    this.updateVertices();
  }

  updateVertices() {
    const noseRad = this.noseAngle * Math.PI / 180;
    const rearRad = Math.max(this.rearAngle * Math.PI / 180, 0.001);
    const noseX = this.height / Math.tan(noseRad);
    const rearX = this.height / Math.tan(rearRad);
    this.vertices = [
      new Vector2(0, 0),
      new Vector2(this.length, 0),
      new Vector2(this.length - rearX, this.height),
      new Vector2(noseX, this.height)
    ];
  }

  getAbsoluteVertices(carX, carY) {
    const offset = new Vector2(carX - this.length / 2, carY - this.height / 2);
    return this.vertices.map(v => v.add(offset));
  }
}

class Particle {
  constructor() {
    this.pos = new Vector2(0, 0);
    this.vel = new Vector2(0, 0);
    this.age = 0;
    this.maxAge = 900;
    this.tail = [];
    this.minDist = Infinity;
    this.minSpeed = Infinity;
    this.maxAbsVelY = 0;
  }

  update(sim) {
    // Find closest point on car edges
    let minDistSq = Infinity;
    let repelDir = new Vector2(0, 0);
    const verts = sim.car.getAbsoluteVertices(sim.carX, sim.carY);
    for (let i = 0; i < verts.length; i++) {
      const p1 = verts[i];
      const p2 = verts[(i + 1) % verts.length];
      const tang = p2.sub(p1);
      const toPos = this.pos.sub(p1);
      let t = toPos.dot(tang) / tang.magSq();
      t = Math.max(0, Math.min(1, t));
      const closest = p1.add(tang.scale(t));
      const delta = this.pos.sub(closest);
      const distSq = delta.magSq();
      if (distSq < minDistSq) {
        minDistSq = distSq;
        repelDir = delta.normalize();
      }
    }

    const minDist = Math.sqrt(minDistSq);
    this.minDist = Math.min(this.minDist, minDist);

    // Compute new velocity statelessly
    let perturb = new Vector2(0, 0);
    if (minDistSq < 2500) {
      const strength = 500 / (minDist * minDist);
      perturb = repelDir.scale(strength);
    }

    let vel = new Vector2(sim.windSpeed, 0).add(perturb);

    const turbX = (Math.random() - 0.5) * sim.turbulence;
    const turbY = (Math.random() - 0.5) * sim.turbulence;
    vel = vel.add(new Vector2(turbX, turbY));

    if (minDist < 60) {
      vel = vel.scale(minDist / 60);
    }

    // Track metrics over lifespan
    this.minSpeed = Math.min(this.minSpeed, vel.mag());
    this.maxAbsVelY = Math.max(this.maxAbsVelY, Math.abs(vel.y));

    // Update position
    this.pos = this.pos.add(vel);
    this.age++;

    // Update tail
    this.tail.push(this.pos.clone());
    if (this.tail.length > 20) {
      this.tail.shift();
    }

    // Check expiration
    if (this.pos.x > sim.canvasW + 50 || this.age > this.maxAge) {
      if (this.minDist < 40) {
        const loss = 1 - (this.minSpeed / sim.windSpeed);
        if (loss > 0.1) {
          sim.dragSum += loss;
          sim.liftSum += this.maxAbsVelY / sim.windSpeed;
          sim.deflCount++;
        }
      }
      this.reset(sim);
    }
  }

  reset(sim) {
    this.pos = new Vector2(-30 - Math.random() * 20, sim.canvasH * 0.2 + Math.random() * (sim.canvasH * 0.6));
    this.vel = new Vector2(0, 0);
    this.age = 0;
    this.tail = [];
    this.minDist = Infinity;
    this.minSpeed = Infinity;
    this.maxAbsVelY = 0;
  }
}

class Simulation {
  constructor(canvas) {
    this.car = new Car();
    this.particles = [];
    this.windSpeed = 2;
    this.turbulence = 0.1;
    this.paused = false;
    this.showStreamlines = true;
    this.showMetrics = true;
    this.particleCount = 300;
    this.dragCoeff = 0;
    this.liftCoeff = 0;
    this.dragSum = 0;
    this.liftSum = 0;
    this.deflCount = 0;
    this.canvasW = canvas.width;
    this.canvasH = canvas.height;
    this.carX = this.canvasW * 0.5;
    this.carY = this.canvasH * 0.5;
  }

  init() {
    this.resetParticles();
  }

  resetParticles() {
    this.dragSum = 0;
    this.liftSum = 0;
    this.deflCount = 0;
    this.particles = [];
    for (let i = 0; i < this.particleCount; i++) {
      const p = new Particle();
      p.reset(this);
      this.particles.push(p);
    }
  }

  update() {
    if (!this.paused) {
      for (let p of this.particles) {
        p.update(this);
      }
      if (this.deflCount > 0) {
        this.dragCoeff = this.dragSum / this.deflCount;
        this.liftCoeff = this.liftSum / this.deflCount;
      }
    }
  }

  draw(ctx) {
    ctx.clearRect(0, 0, this.canvasW, this.canvasH);

    // Draw car
    const carVerts = this.car.getAbsoluteVertices(this.carX, this.carY);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(carVerts[0].x, carVerts[0].y);
    for (let i = 1; i < carVerts.length; i++) {
      ctx.lineTo(carVerts[i].x, carVerts[i].y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = 'rgba(100, 100, 200, 0.3)';
    ctx.fill();

    // Draw particles and streamlines
    for (let p of this.particles) {
      const alpha = Math.max(0, 1 - p.age / p.maxAge);
      if (alpha <= 0) continue;

      const speed = p.vel.mag();
      const hue = Math.max(0, 240 - speed * 15);
      const color = `hsla(${hue}, 100%, 50%, ${alpha})`;

      // Particle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p.pos.x, p.pos.y, Math.max(1.5, 3 - speed * 0.1), 0, 2 * Math.PI);
      ctx.fill();

      // Streamline tail
      if (this.showStreamlines && p.tail.length > 1) {
        ctx.strokeStyle = `hsla(${hue}, 100%, 50%, ${alpha * 0.4})`;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(p.tail[0].x, p.tail[0].y);
        for (let j = 1; j < p.tail.length; j++) {
          ctx.lineTo(p.tail[j].x, p.tail[j].y);
        }
        ctx.stroke();
      }
    }
  }
}

let canvas, ctx, sim, metricsEl;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = Math.min(1200, rect.width);
  const h = w / 2;
  canvas.width = w;
  canvas.height = h;
  sim.canvasW = w;
  sim.canvasH = h;
  sim.carX = w * 0.5;
  sim.carY = h * 0.5;
  sim.resetParticles();
}

function onParamChange(param, value) {
  switch (param) {
    case 'length':
      sim.car.length = value;
      break;
    case 'height':
      sim.car.height = value;
      break;
    case 'noseAngle':
      sim.car.noseAngle = value;
      break;
    case 'rearAngle':
      sim.car.rearAngle = value;
      break;
    case 'windSpeed':
      sim.windSpeed = value;
      break;
    case 'turbulence':
      sim.turbulence = value;
      break;
    case 'particleCount':
      sim.particleCount = Math.round(value);
      sim.resetParticles();
      return;
    case 'showStreamlines':
      sim.showStreamlines = !!value;
      return;
    case 'showMetrics':
      sim.showMetrics = !!value;
      return;
  }
  if (['length', 'height', 'noseAngle', 'rearAngle'].includes(param)) {
    sim.car.updateVertices();
    sim.resetParticles();
  }
}

function animationLoop() {
  sim.update();
  sim.draw(ctx);
  if (sim.showMetrics) {
    metricsEl.innerHTML = `Drag: ${sim.dragCoeff.toFixed(3)}<br>Lift: ${sim.liftCoeff.toFixed(3)}`;
  } else {
    metricsEl.innerHTML = '';
  }
  requestAnimationFrame(animationLoop);
}

function initApp() {
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
  canvas.width = 1200;
  canvas.height = 600;
  resize();
  sim = new Simulation(canvas);
  metricsEl = document.getElementById('metrics');

  const params = ['length', 'height', 'noseAngle', 'rearAngle', 'windSpeed', 'particleCount', 'turbulence'];
  params.forEach(param => {
    const slider = document.getElementById(param);
    const numInput = document.getElementById(param + '_num');
    const valSpan = document.getElementById(param + '_val');

    const updateParam = (e) => {
      let v = e ? parseFloat(e.target.value) : parseFloat(slider.value);
      slider.value = v;
      numInput.value = v;
      const fixed = param === 'turbulence' ? 2 : param === 'windSpeed' ? 1 : 0;
      valSpan.textContent = v.toFixed(fixed);
      onParamChange(param, v);
    };

    slider.oninput = updateParam;
    numInput.onchange = updateParam;

    // Initial sync
    updateParam();
  });

  // Toggles
  document.getElementById('showStreamlines').onchange = (e) => onParamChange('showStreamlines', e.target.checked);
  document.getElementById('showMetrics').onchange = (e) => onParamChange('showMetrics', e.target.checked);

  // Buttons
  const pauseBtn = document.getElementById('pauseBtn');
  pauseBtn.onclick = () => {
    sim.paused = !sim.paused;
    pauseBtn.textContent = sim.paused ? 'Resume' : 'Pause';
  };
  document.getElementById('resetBtn').onclick = () => sim.resetParticles();

  window.addEventListener('resize', resize);

  sim.init();
  animationLoop();
}