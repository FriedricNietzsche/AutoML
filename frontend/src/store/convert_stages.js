// Read projectStore.ts and convert stages object to array before merging
const fs = require('fs');
const content = fs.readFileSync('projectStore.ts', 'utf8');

const newHydrate = content.replace(
  /const json = \(await res\.json\(\)\) as ProjectSnapshot;[\s\S]*?const incomingStages = mergeStages\(get\(\)\.stages, json\.stages\);/,
  `const json = (await res.json()) as ProjectSnapshot;
      // Convert stages from object to array if needed
      const stagesArray = json.stages && typeof json.stages === 'object' && !Array.isArray(json.stages)
        ? Object.values(json.stages)
        : json.stages;
      const incomingStages = mergeStages(get().stages, stagesArray);`
);

const newConfirm = newHydrate.replace(
  /const json = \(await res\.json\(\)\) as ProjectSnapshot;[\s\S]*?set\(\{[\s\S]*?stages: mergeStages\(get\(\)\.stages, json\.stages\),/,
  `const json = (await res.json()) as ProjectSnapshot;
      // Convert stages from object to array if needed
      const stagesArray2 = json.stages && typeof json.stages === 'object' && !Array.isArray(json.stages)
        ? Object.values(json.stages)
        : json.stages;
      set({
        stages: mergeStages(get().stages, stagesArray2),`
);

fs.writeFileSync('projectStore.ts', newConfirm, 'utf8');
console.log('✓ Fixed hydrate and confirm to convert stages object to array');
