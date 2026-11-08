// V-World 3D 지도(WebGL 3.0) 위에서 시뮬레이터 프레임을 찍는다.
//
// index.html 의 shot / sweep 과 같은 자리, 같은 메타 형식으로 찍는다. 다른 점은
//   - 지도와 3D 건물이 V-World 것이다(사진 질감 건물)
//   - 연기는 smoke.js 로 그린다(V-World 가 장면 시계를 멈춰 Cesium 입자가 안 움직임)
//   - 지형 높이는 V-World 지형(해발 기준)에서 읽는다. 다 불러올 때까지 기다린다
//   - 건물 이름 같은 지도 글자는 끈다(탐지가 글자를 연기로 볼 수 있다)
//
//     await vwPrepare(FIRE)                 // 글자 끄기, 연기 피우기
//     await vwSweep([200, 300, 400, 600, 800], 120, 0)
//
// 프레임은 tools/sim_server.py 의 POST /frame?site=jagok3d 로 sim/frames_jagok3d/ 에 저장된다.
(function () {
  const C = Cesium;
  const wait = ms => new Promise(r => setTimeout(r, ms));

  function ground(lon, lat) {
    const h = viewer.scene.globe.getHeight(C.Cartographic.fromDegrees(lon, lat));
    return h === undefined ? null : h;
  }

  // 지형과 3D 건물 타일이 다 뜰 때까지 그린다. 높이는 두 번 연속 같아야 믿는다
  async function settleAt(lon, lat, maxMs) {
    const end = Date.now() + (maxMs || 25000);
    let prev = null, same = 0;
    while (Date.now() < end) {
      viewer.scene.render();
      await wait(250);
      const tiles = [];
      const P = viewer.scene.primitives;
      for (let i = 0; i < P.length; i++) {
        const p = P.get(i);
        if (p instanceof C.Cesium3DTileset && p.show) tiles.push(p.tilesLoaded);
      }
      const h = ground(lon, lat);
      same = (h !== null && prev !== null && Math.abs(h - prev) < 0.05) ? same + 1 : 0;
      prev = h;
      if (viewer.scene.globe.tilesLoaded && tiles.every(Boolean) && same >= 4) return h;
    }
    return prev;
  }

  // 지도 글자(건물 이름, 동 이름)를 끈다. 글자는 POI 3D 타일셋(POI_BASE, POI_BOUND)과
  // 글자 모음으로 그려진다. 우리 연기 빌보드(smoke.collection)는 남긴다
  function hideLabels() {
    const keep = new Set((typeof lineSmokes !== "undefined" && lineSmokes.length ? lineSmokes : (smoke ? [smoke] : []))
      .map(s => s.collection));
    const P = viewer.scene.primitives;
    let n = 0;
    for (let i = 0; i < P.length; i++) {
      const p = P.get(i);
      if (keep.has(p)) continue;
      if (p instanceof C.Cesium3DTileset) {
        const url = String((p.resource && p.resource.url) || "");
        if (url.indexOf("/poi/") >= 0) { p.show = false; n++; }
        continue;
      }
      if (p instanceof C.LabelCollection || p instanceof C.BillboardCollection) { p.show = false; n++; }
    }
    viewer.entities.values.forEach(e => { if (e.label || e.billboard) { e.show = false; n++; } });
    return n;
  }

  function cameraBlock() {
    const cam = viewer.camera;
    const c = C.Cartographic.fromCartesian(cam.positionWC);
    const g = ground(C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude));
    return {
      lat: +C.Math.toDegrees(c.latitude).toFixed(7),
      lon: +C.Math.toDegrees(c.longitude).toFixed(7),
      alt_m: g === null ? null : +(c.height - g).toFixed(1),
      alt_ref: "지면",
      heading_deg: +((C.Math.toDegrees(cam.heading) % 360 + 360) % 360).toFixed(1),
      pitch_deg: +C.Math.toDegrees(cam.pitch).toFixed(1),
      hfov_deg: +C.Math.toDegrees(cam.frustum.fov).toFixed(1),
      source: "가상",
    };
  }

  let FIRE = null, smoke = null, fireGround = null;

  window.vwPrepare = async function (fire, growSec) {
    FIRE = fire;
    viewer.camera.frustum.fov = C.Math.toRadians(60);
    viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(fire.lon, fire.lat, 1200),
      orientation: { heading: 0, pitch: C.Math.toRadians(-60), roll: 0 } });
    await settleAt(fire.lon, fire.lat, 30000);
    // 낮게 내려가야 가장 세밀한 지형이 뜬다
    viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(fire.lon, fire.lat - 0.003, 300),
      orientation: { heading: 0, pitch: C.Math.toRadians(-40), roll: 0 } });
    fireGround = await settleAt(fire.lon, fire.lat, 30000);
    if (smoke) smoke.remove();
    smoke = makeSmoke(viewer, fire.lon, fire.lat, fireGround, 2);
    smoke.warm(growSec === undefined ? 90 : growSec);
    window.__smoke = smoke;
    return { fireGround, labelsHidden: hideLabels() };
  };

  // 번진 산불: 선을 따라 연기를 여러 개 피운다. 카메라는 선의 가운데(첫 인자 center)를 겨냥하고,
  // 프레임 메타의 truth.line 에 연기 자리 전부를 적는다.
  //
  //     await vwPrepareLine({lon, lat}, [{lon, lat}, ...], 90)
  let LINE = null, lineSmokes = [];
  window.vwPrepareLine = async function (center, points, growSec) {
    const r = await vwPrepare(center, 0);
    if (smoke) { smoke.remove(); smoke = null; }
    lineSmokes.forEach(s => s.remove());
    lineSmokes = [];
    const carto = points.map(p => C.Cartographic.fromDegrees(p.lon, p.lat));
    await C.sampleTerrain(viewer.terrainProvider, 15, carto);
    points.forEach((p, i) => {
      const s = makeSmoke(viewer, p.lon, p.lat, carto[i].height, 2);
      s.warm(growSec === undefined ? 90 : growSec);
      lineSmokes.push(s);
    });
    smoke = lineSmokes[0];                // hideLabels 가 남길 것 (아래에서 전부 남기게 고침)
    LINE = points.map((p, i) => ({ lat: p.lat, lon: p.lon, ground_m: +carto[i].height.toFixed(1) }));
    window.__lineSmokes = lineSmokes;
    return Object.assign(r, { sources: points.length });
  };

  window.vwShot = async function (D, h, bearing) {
    const T = FIRE;
    const B = C.Math.toRadians(bearing);
    const lat = T.lat + (D * Math.cos(B)) / 111320;
    const lon = T.lon + (D * Math.sin(B)) / (111320 * Math.cos(C.Math.toRadians(lat)));
    const heading = (bearing + 180) % 360;
    // 먼저 위에서 그 자리 타일을 띄워 지면 높이를 읽는다
    viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(lon, lat, 400),
      orientation: { heading: C.Math.toRadians(heading), pitch: C.Math.toRadians(-60), roll: 0 } });
    const g = await settleAt(lon, lat, 20000);
    if (g === null) return { ok: false, why: "지형 높이를 못 읽음" };
    const pitch = -Math.atan2(h, D) * 180 / Math.PI;
    viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(lon, lat, g + h),
      orientation: { heading: C.Math.toRadians(heading), pitch: C.Math.toRadians(pitch), roll: 0 } });
    await settleAt(T.lon, T.lat, 25000);
    hideLabels();
    for (let i = 0; i < 6; i++) { viewer.scene.render(); await wait(60); }
    viewer.scene.render();
    const png = viewer.canvas.toDataURL("image/png");
    const meta = {
      schema_version: "0.3", video_id: "SWEEP_VW3D", frame_id: Date.now() % 100000,
      time: new Date().toISOString(),
      image_size: { width: viewer.canvas.width, height: viewer.canvas.height },
      camera: cameraBlock(), detections: [],
      truth: LINE ? { fire_lat: T.lat, fire_lon: T.lon, line: LINE } : { fire_lat: T.lat, fire_lon: T.lon },
      sweep: { D: D, h: h, bearing: bearing, ground_m: +g.toFixed(1), fire_ground_m: +fireGround.toFixed(1),
               terrain: "V-World (해발)", buildings: "V-World 3D" },
    };
    const res = await fetch("/frame?site=" + (window.__shotSite || "jagok3d"), { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ png: png, meta: meta }) });
    return { ok: true, saved: (await res.json()).saved, D, bearing, ground: g };
  };

  window.vwSweep = async function (dists, h, bearing) {
    const out = [];
    for (const D of dists) out.push(await vwShot(D, h === undefined ? 120 : h, bearing));
    return out;
  };

  // 드론이 발화점을 끼고 원을 그리며 나는 장면을 한 장씩 찍는다(시연 영상용).
  // 장면 시간은 fps 로 고정해서 흐르게 한다. 컴퓨터가 느려도 영상 속 속도는 같다.
  // 처음 한 바퀴는 찍지 않고 날아서 타일을 받아 둔다(warm).
  //
  //     await vwFlight({ radius: 350, agl: 120, sec: 24, fps: 12, turns: 1 })
  //     -> sim/frames_flight/sim_0000.png ...  tools/frames_to_video.py 로 mp4
  window.vwFlight = async function (o) {
    o = Object.assign({ radius: 350, agl: 120, sec: 24, fps: 12, turns: 1, start: 200, warm: true }, o || {});
    const T = FIRE;
    const n = Math.round(o.sec * o.fps);
    const mLat = 111320, mLon = 111320 * Math.cos(C.Math.toRadians(T.lat));
    const pose = k => {
      const bearing = o.start + 360 * o.turns * k / n;              // 드론이 발화점의 어느 쪽에 있나
      const B = C.Math.toRadians(bearing);
      const lat = T.lat + o.radius * Math.cos(B) / mLat, lon = T.lon + o.radius * Math.sin(B) / mLon;
      return { lat, lon, heading: (bearing + 180) % 360 };
    };
    const place = (p, g) => {
      const pitch = -Math.atan2(o.agl + g - fireGround, o.radius) * 180 / Math.PI;
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, g + o.agl),
        orientation: { heading: C.Math.toRadians(p.heading), pitch: C.Math.toRadians(pitch), roll: 0 } });
    };
    // 지면 높이는 V-World 지형에서 한 번에 읽는다(장마다 화면을 기다리지 않게)
    const P = [];
    for (let k = 0; k < n; k++) P.push(pose(k));
    const carto = P.map(p => C.Cartographic.fromDegrees(p.lon, p.lat));
    await C.sampleTerrain(viewer.terrainProvider, 15, carto);
    const G = carto.map(c => c.height || fireGround);
    if (o.warm) {
      for (let k = 0; k < n; k += Math.max(1, Math.round(o.fps))) { place(P[k], G[k]); await settleAt(T.lon, T.lat, 6000); }
    }
    const saved = [];
    smoke.setManual(true);
    for (let k = 0; k < n; k++) {
      place(P[k], G[k]);
      smoke.step(1 / o.fps);
      await settleAt(T.lon, T.lat, 1500);
      hideLabels();
      viewer.scene.render();
      const res = await fetch("/frame?site=flight", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ png: viewer.canvas.toDataURL("image/png"),
                               meta: { k: k, fps: o.fps, camera: cameraBlock(), truth: { fire_lat: T.lat, fire_lon: T.lon } } }) });
      saved.push((await res.json()).saved);
      window.__flightK = k;
    }
    smoke.setManual(false);
    return { frames: saved.length };
  };

  // 시연 장면: 드론이 멀리서 연기 쪽으로 곧장 다가온다. 1초(장면 시간)마다 한 장씩
  // 탐지용 프레임을 찍는다. 메타는 vwShot 과 같은 형식이라 파이프라인에 그대로 넣는다.
  //
  //     await vwApproach({ from: 1000, to: 200, speed: 15, bearing: 200, agl: 120 })
  //     -> sim/frames_approach/sim_0000.png ...
  window.vwApproach = async function (o) {
    o = Object.assign({ from: 1000, to: 200, speed: 15, bearing: 200, agl: 120, every: 1.0, site: "approach" }, o || {});
    const T = FIRE;
    const mLat = 111320, mLon = 111320 * Math.cos(C.Math.toRadians(T.lat));
    const B = C.Math.toRadians(o.bearing), heading = (o.bearing + 180) % 360;
    const n = Math.floor((o.from - o.to) / (o.speed * o.every)) + 1;
    const pts = [];
    for (let k = 0; k < n; k++) {
      const D = o.from - k * o.speed * o.every;
      pts.push({ D, lat: T.lat + D * Math.cos(B) / mLat, lon: T.lon + D * Math.sin(B) / mLon });
    }
    const carto = pts.map(p => C.Cartographic.fromDegrees(p.lon, p.lat));
    await C.sampleTerrain(viewer.terrainProvider, 15, carto);
    smoke.setManual(true);
    const saved = [];
    for (let k = 0; k < n; k++) {
      const p = pts[k], g = carto[k].height;
      const pitch = -Math.atan2(o.agl + g - fireGround, p.D) * 180 / Math.PI;
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, g + o.agl),
        orientation: { heading: C.Math.toRadians(heading), pitch: C.Math.toRadians(pitch), roll: 0 } });
      smoke.step(o.every);
      await settleAt(T.lon, T.lat, 12000);
      hideLabels();
      for (let i = 0; i < 4; i++) { viewer.scene.render(); await wait(40); }
      viewer.scene.render();
      const meta = {
        schema_version: "0.3", video_id: "APPROACH_VW3D", frame_id: k,
        time: new Date(Date.UTC(2026, 10, 6, 3, 0, 0) + k * o.every * 1000).toISOString(),
        image_size: { width: viewer.canvas.width, height: viewer.canvas.height },
        camera: cameraBlock(), detections: [],
        truth: { fire_lat: T.lat, fire_lon: T.lon },
        sweep: { D: p.D, h: o.agl, bearing: o.bearing, ground_m: +g.toFixed(1), fire_ground_m: +fireGround.toFixed(1),
                 t_s: k * o.every, terrain: "V-World (해발)", buildings: "V-World 3D" },
      };
      const res = await fetch("/frame?site=" + o.site, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ png: viewer.canvas.toDataURL("image/png"), meta: meta }) });
      saved.push((await res.json()).saved);
      window.__approachK = k;
    }
    smoke.setManual(false);
    return { frames: saved.length };
  };

  // 신고 시나리오. 드론은 불 위치를 모른다. 카메라는 신고 지점이나 시스템이 추정한
  // 위치만 겨냥하고, 불 위치(FIRE)는 채점용 truth 에만 적는다.
  async function saveFrame(site, videoId, k, t_s, extra) {
    hideLabels();
    for (let i = 0; i < 4; i++) { viewer.scene.render(); await wait(40); }
    viewer.scene.render();
    const meta = Object.assign({
      schema_version: "0.3", video_id: videoId, frame_id: k,
      time: new Date(Date.UTC(2026, 10, 7, 3, 0, 0) + t_s * 1000).toISOString(),
      image_size: { width: viewer.canvas.width, height: viewer.canvas.height },
      camera: cameraBlock(), detections: [],
      truth: { fire_lat: FIRE.lat, fire_lon: FIRE.lon },
    }, extra);
    const res = await fetch("/frame?site=" + site, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ png: viewer.canvas.toDataURL("image/png"), meta: meta }) });
    return (await res.json()).saved;
  }

  // 수색 비행: 신고 지점(center)을 가운데 두고 원을 그리며, 카메라는 신고 지점을 본다.
  //     await vwSearch({ center: {lat, lon}, radius: 600, agl: 150, stepDeg: 10, start: 180, speed: 15 })
  window.vwSearch = async function (o) {
    o = Object.assign({ radius: 600, agl: 150, stepDeg: 10, start: 180, turns: 1, speed: 15, site: "search" }, o || {});
    const c = o.center, mLat = 111320, mLon = 111320 * Math.cos(C.Math.toRadians(c.lat));
    const n = Math.round(360 * o.turns / o.stepDeg);
    const P = [];
    for (let k = 0; k < n; k++) {
      const b = o.start + k * o.stepDeg, B = C.Math.toRadians(b);
      P.push({ b, lat: c.lat + o.radius * Math.cos(B) / mLat, lon: c.lon + o.radius * Math.sin(B) / mLon });
    }
    const carto = P.map(p => C.Cartographic.fromDegrees(p.lon, p.lat)).concat([C.Cartographic.fromDegrees(c.lon, c.lat)]);
    await C.sampleTerrain(viewer.terrainProvider, 15, carto);
    const gC = carto[n].height;
    const dt = (2 * Math.PI * o.radius * o.stepDeg / 360) / o.speed;   // 장면 시간(초)
    smoke.setManual(true);
    const saved = [];
    for (let k = 0; k < n; k++) {
      const p = P[k], g = carto[k].height;
      const pitch = -Math.atan2(o.agl + g - gC, o.radius) * 180 / Math.PI;
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, g + o.agl),
        orientation: { heading: C.Math.toRadians((p.b + 180) % 360), pitch: C.Math.toRadians(pitch), roll: 0 } });
      if (k) smoke.step(dt);
      await settleAt(c.lon, c.lat, 12000);
      saved.push(await saveFrame(o.site, "SEARCH_VW3D", k, k * dt,
        { search: { center: c, radius: o.radius, bearing: p.b, agl: o.agl, t_s: +(k * dt).toFixed(1) } }));
      window.__searchK = k;
    }
    smoke.setManual(false);
    return { frames: saved.length, dt, t_end: +(n * dt).toFixed(1) };
  };

  // 추정 위치로 접근: from(드론 자리)에서 target(시스템이 추정한 발화점)으로 곧장 날며
  // target 을 본다. target 에서 stop m 앞에서 멈춘다.
  //     await vwApproachTo({ from: {lat, lon}, target: {lat, lon}, agl: 150, speed: 15, stop: 150, t0: 300 })
  window.vwApproachTo = async function (o) {
    o = Object.assign({ agl: 150, speed: 15, every: 1.0, stop: 150, t0: 0, site: "approachto" }, o || {});
    const f = o.from, t = o.target, mLat = 111320, mLon = 111320 * Math.cos(C.Math.toRadians(t.lat));
    const dE = (f.lon - t.lon) * mLon, dN = (f.lat - t.lat) * mLat;
    const D0 = Math.hypot(dE, dN), bearing = (Math.atan2(dE, dN) * 180 / Math.PI + 360) % 360;
    const B = C.Math.toRadians(bearing), heading = (bearing + 180) % 360;
    const n = Math.max(1, Math.floor((D0 - o.stop) / (o.speed * o.every)) + 1);
    const P = [];
    for (let k = 0; k < n; k++) {
      const D = D0 - k * o.speed * o.every;
      P.push({ D, lat: t.lat + D * Math.cos(B) / mLat, lon: t.lon + D * Math.sin(B) / mLon });
    }
    const carto = P.map(p => C.Cartographic.fromDegrees(p.lon, p.lat)).concat([C.Cartographic.fromDegrees(t.lon, t.lat)]);
    await C.sampleTerrain(viewer.terrainProvider, 15, carto);
    const gT = carto[n].height;
    smoke.setManual(true);
    const saved = [];
    for (let k = 0; k < n; k++) {
      const p = P[k], g = carto[k].height;
      const pitch = -Math.atan2(o.agl + g - gT, p.D) * 180 / Math.PI;
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, g + o.agl),
        orientation: { heading: C.Math.toRadians(heading), pitch: C.Math.toRadians(pitch), roll: 0 } });
      smoke.step(o.every);
      await settleAt(t.lon, t.lat, 12000);
      const tt = o.t0 + k * o.every;
      saved.push(await saveFrame(o.site, "APPROACHTO_VW3D", k, tt,
        { approach: { target: t, D_to_target: +p.D.toFixed(1), bearing: +bearing.toFixed(1), agl: o.agl, t_s: +tt.toFixed(1) } }));
      window.__approachK = k;
    }
    smoke.setManual(false);
    return { frames: saved.length, D0: +D0.toFixed(1), bearing: +bearing.toFixed(1) };
  };

  // 정답 계산용 지형 격자. V-World 지형을 화면에 띄워 읽으므로 느리다. 칸마다 카메라를 옮긴다
  window.vwHeights = async function (pts) {
    const out = [];
    for (const p of pts) {
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(p.lon, p.lat, 400),
        orientation: { heading: 0, pitch: C.Math.toRadians(-90), roll: 0 } });
      out.push(await settleAt(p.lon, p.lat, 15000));
    }
    return out;
  };
})();
