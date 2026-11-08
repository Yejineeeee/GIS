// 연기를 입자 대신 빌보드로 직접 움직인다.
//
// Cesium 의 ParticleSystem 은 장면 시계로 입자 나이를 센다. V-World 3D 지도(WebGL 3.0)는
// 시계를 멈춰 두므로 입자가 나이를 먹지 않고 발화점에 흰 반구로만 쌓였다. 그래서 시간을
// 우리가 performance.now() 로 재고, 빌보드 위치와 크기, 투명도를 매 화면마다 고친다.
//
// 모양 값은 index.html 의 lightSmoke 와 같게 맞췄다(알파 0.12, 크기 8m x 2 -> 13배, 수명
// 11~24초, 속도 3~13m/s, 원뿔 20도, 초당 240개, 높이에 따라 세지는 바람).
//
//     const s = makeSmoke(viewer, lon, lat, groundHeight);   // 지면 높이(타원체든 해발이든 그 뷰어 기준)
//     s.step(dt)       // 직접 시간을 흘릴 때. 보통은 preRender 에 걸려 저절로 돈다
//     s.remove()
(function () {
  function puffDataUrl() {
    const n = 128, c = document.createElement("canvas");
    c.width = c.height = n;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(n, n);
    const noise = (x, y, f) =>
      Math.sin(x * f * 1.7 + Math.cos(y * f * 2.3) * 2.1) *
      Math.cos(y * f * 1.3 + Math.sin(x * f * 1.9) * 1.7);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const dx = (x - n / 2) / (n / 2), dy = (y - n / 2) / (n / 2);
      const r = Math.hypot(dx, dy);
      let v = 0.55 * noise(x, y, 0.06) + 0.3 * noise(x, y, 0.13) + 0.15 * noise(x, y, 0.27);
      v = (v + 1) / 2;
      const fall = Math.max(0, 1 - r);
      const a = Math.pow(fall, 1.6) * (0.55 + 0.45 * v);
      const i = (y * n + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = 245;
      img.data[i + 3] = Math.round(255 * Math.min(1, a));
    }
    ctx.putImageData(img, 0, 0);
    return c.toDataURL();
  }

  const P = { alpha0: 0.12, size: 8, scale0: 2.0, scale1: 13.0, life0: 11, life1: 24,
              speed0: 3, speed1: 13, cone: 20 * Math.PI / 180, rate: 240 };

  window.makeSmoke = function (viewer, lon, lat, ground, agl) {
    const C = Cesium;
    const base = ground + (agl === undefined ? 2 : agl);
    const origin = C.Cartesian3.fromDegrees(lon, lat, base);
    const enu = C.Transforms.eastNorthUpToFixedFrame(origin);
    const rot = C.Matrix4.getMatrix3(enu, new C.Matrix3());
    const coll = viewer.scene.primitives.add(new C.BillboardCollection({ scene: viewer.scene }));
    const image = puffDataUrl();
    const grey = C.Color.fromCssColorString("#c8c8c8");
    const parts = [];
    let carry = 0, last = performance.now(), dead = false, manual = false;

    function spawn() {
      const th = Math.random() * 2 * Math.PI, ph = Math.random() * P.cone;
      const sp = P.speed0 + Math.random() * (P.speed1 - P.speed0);
      const p = {
        e: 0, n: 0, u: 0,                                   // 발화점 기준 동, 북, 위 (m)
        ve: sp * Math.sin(ph) * Math.cos(th), vn: sp * Math.sin(ph) * Math.sin(th), vu: sp * Math.cos(ph),
        age: 0, life: P.life0 + Math.random() * (P.life1 - P.life0),
        j: [Math.random() - 0.5, Math.random() - 0.5],
        b: coll.add({ position: origin, image: image, sizeInMeters: true,
                      width: P.size * P.scale0, height: P.size * P.scale0,
                      color: C.Color.WHITE.withAlpha(P.alpha0) }),
      };
      parts.push(p);
    }

    const local = new C.Cartesian3(), world = new C.Cartesian3();
    function step(dt) {
      if (dead) return;
      carry += P.rate * dt;
      while (carry >= 1) { spawn(); carry -= 1; }
      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        p.age += dt;
        if (p.age >= p.life) { coll.remove(p.b); parts.splice(i, 1); continue; }
        // 높이에 따라 바람이 세지고(바람 전단) 옆으로 더 흩어진다 (index.html 과 같은 식)
        const k = Math.min(Math.max(0, p.u) / 80, 1);
        const wind = 1.2 + 6.5 * k, spread = 1.0 + 5.0 * k;
        p.e += (p.ve + wind * 0.9 + p.j[0] * spread) * dt;
        p.n += (p.vn + wind * 0.42 + p.j[1] * spread) * dt;
        p.u += p.vu * dt;
        local.x = p.e; local.y = p.n; local.z = p.u;
        C.Matrix3.multiplyByVector(rot, local, world);
        C.Cartesian3.add(origin, world, world);
        p.b.position = world;
        const t = p.age / p.life;
        const s = P.size * (P.scale0 + (P.scale1 - P.scale0) * t);
        p.b.width = s; p.b.height = s;
        p.b.color = C.Color.lerp(C.Color.WHITE, grey, t, new C.Color()).withAlpha(P.alpha0 * (1 - t));
      }
    }

    function onPre() {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);       // 창이 뒤로 갔다 오면 한 번에 뛰지 않게
      last = now;
      if (!manual) step(dt);                               // 영상을 찍을 때는 step() 으로만 흘린다
    }
    viewer.scene.preRender.addEventListener(onPre);

    return {
      collection: coll,
      setManual(v) { manual = !!v; },
      step: step,
      count: () => parts.length,
      remove() { dead = true; viewer.scene.preRender.removeEventListener(onPre); viewer.scene.primitives.remove(coll); },
      // 기둥이 다 자랄 때까지 시간을 직접 흘린다. 화면을 그리지 않아도 된다
      warm(sec, dt) { dt = dt || 0.1; for (let t = 0; t < sec; t += dt) step(dt); last = performance.now(); },
    };
  };
})();
