# %%
# Task 4: run after the original Task 3 cells. Reuses LeafDataset and LeafNetBN.
import csv
import hashlib
import random

task4_dir = Path('task4_round2')
task4_dir.mkdir(exist_ok=True)

# Reuse the EXACT camera set and split from round 1, even in a new runtime.
old_dir = Path('task4_results')
if (old_dir / 'camera_validation.pt').exists() and (old_dir / 'split.json').exists():
    old_camera = (old_dir / 'camera_validation.pt').read_bytes()
    old_split = (old_dir / 'split.json').read_bytes()
else:
    if not Path('task4_results.zip').exists():
        print('Choose your original task4_results.zip (the 12 completed runs).')
        from google.colab import files
        files.upload()
    with zipfile.ZipFile('task4_results.zip') as previous:
        old_camera = previous.read('camera_validation.pt')
        old_split = previous.read('split.json')
assert hashlib.sha256(old_camera).hexdigest() == '3b4cbe291f15cad9342b6a9276d2acf705501ea0c1a4e0762bc51a01425ed9da', 'Use the original round-1 results ZIP.'
(task4_dir / 'camera_validation.pt').write_bytes(old_camera)
(task4_dir / 'split.json').write_bytes(old_split)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Read the same split, without changing any original datasets or transforms.
plain_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean, std)
])
task4_train = LeafDataset('data/leaves/train', plain_transform)
task4_val = LeafDataset('data/leaves/val', plain_transform)
assert task4_train.classes == task4_val.classes == ['apple', 'cherry', 'chestnut', 'maple']
assert len(task4_train) > 0 and len(task4_val) > 0
print('train:', len(task4_train), '| val:', len(task4_val))
for label, name in enumerate(task4_train.classes):
    print(name, 'train:', sum(y == label for _, y in task4_train.samples),
          'val:', sum(y == label for _, y in task4_val.samples))
print('One validation image is worth', 100 / len(task4_val), 'percentage points.')

# A content manifest checks that reruns use the same images, not just the same counts.
manifest = []
for split, dataset in [('train', task4_train), ('val', task4_val)]:
    for path, label in dataset.samples:
        manifest.append([split, str(Path(path).relative_to('data/leaves')), label,
                         hashlib.sha256(Path(path).read_bytes()).hexdigest()])
manifest_file = task4_dir / 'split.json'
if manifest_file.exists():
    assert json.loads(manifest_file.read_text()) == manifest, 'Dataset changed; use a new results folder.'
else:
    manifest_file.write_text(json.dumps(manifest, indent=2))

# Make the camera-like validation tensors ONCE, then reuse them for every run.
camera_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(), transforms.Normalize(mean, std)
])
camera_file = task4_dir / 'camera_validation.pt'
if camera_file.exists():
    camera_data = torch.load(camera_file, map_location='cpu', weights_only=True)
else:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        camera_images = []
        camera_labels = []
        for path, label in task4_val.samples:
            with Image.open(path) as image:
                camera_images.append(camera_transform(image.convert('RGB')))
            camera_labels.append(label)
    camera_data = {'images': torch.stack(camera_images),
                   'labels': torch.tensor(camera_labels)}
    torch.save(camera_data, camera_file)
assert camera_data['labels'].tolist() == [label for _, label in task4_val.samples]
camera_ds = TensorDataset(camera_data['images'], camera_data['labels'])
task4_val_loader = DataLoader(task4_val, batch_size=32, shuffle=False)
camera_loader = DataLoader(camera_ds, batch_size=32, shuffle=False)
print('Camera validation SHA256:', hashlib.sha256(camera_file.read_bytes()).hexdigest())


# %%
# CNN rows use the Task 3d baseline; MobileNet rows use a separate Task 2 baseline.
# All comparisons use the same images, seeds, optimizer, batch size and final weights.
# name, architecture, geometry, colour, smoothing, cosine, epochs
experiments = [
    ('baseline', 'cnn', True, False, 0.0, False, 15),
    ('cnn_colour_cosine', 'cnn', True, True, 0.0, True, 15),
    ('cnn_smoothing_cosine', 'cnn', True, False, 0.1, True, 15),
    ('cnn_all_three', 'cnn', True, True, 0.1, True, 15),
    ('mobile_full', 'mobile_full', False, False, 0.0, False, 15),
    ('mobile_frozen', 'mobile_frozen', False, False, 0.0, False, 15),
    ('mobile_geometry', 'mobile_full', True, False, 0.0, False, 15),
    ('mobile_colour', 'mobile_full', False, True, 0.0, False, 15),
    ('mobile_cosine', 'mobile_full', False, False, 0.0, True, 15),
    ('mobile_30epochs', 'mobile_full', False, False, 0.0, False, 30),
    ('mobile_combined', 'mobile_full', True, True, 0.0, True, 30),
]
seeds = [0, 1, 2]
records = []
history = []

for name, architecture, geometry, colour, smoothing, cosine, epochs in experiments:
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Same Task 3 geometry; colour jitter is inserted before ToTensor.
        steps = []
        if geometry:
            steps += [transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
                      transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(20)]
        if colour:
            steps.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        run_ds = LeafDataset('data/leaves/train', transforms.Compose(steps))
        assert run_ds.samples == task4_train.samples
        run_loader = DataLoader(run_ds, batch_size=32, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
        if architecture == 'cnn':
            run_model = LeafNetBN()
        else:
            # Same pretrained weights and replacement head initialization for each seed.
            run_model = models.mobilenet_v3_small(weights='DEFAULT')
            run_model.classifier[3] = nn.Linear(run_model.classifier[3].in_features, 4)
            if architecture == 'mobile_frozen':
                for parameter in run_model.parameters():
                    parameter.requires_grad = False
                for parameter in run_model.classifier[3].parameters():
                    parameter.requires_grad = True
        run_model = run_model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        optimizer = torch.optim.Adam((p for p in run_model.parameters() if p.requires_grad), lr=1e-3)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                     if cosine else None)
        started = time.perf_counter()

        # Original Task 3 training loop, with the selected small changes above.
        for epoch in range(epochs):
            run_model.train()
            if architecture == 'mobile_frozen':
                # Freeze BatchNorm statistics and pretrained dropout as well as weights.
                run_model.eval()
                run_model.classifier[3].train()
            loss_sum = 0.0
            learning_rate = optimizer.param_groups[0]['lr']
            for images, labels in run_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(run_model(images), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)

            run_model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in task4_val_loader:
                    images, labels = images.to(device), labels.to(device)
                    correct += (run_model(images).argmax(1) == labels).sum().item()
            history.append({'variant': name, 'seed': seed, 'epoch': epoch + 1,
                            'loss': loss_sum / len(run_ds), 'val_correct': correct,
                            'learning_rate': learning_rate})
            print(f'{name} seed={seed} epoch={epoch+1:2d} '
                  f'loss={loss_sum / len(run_ds):.4f} val={correct}/{len(task4_val)}')
            if scheduler is not None:
                scheduler.step()

        # Same final-epoch selection rule for every variant. No camera-set tuning.
        camera_correct = 0
        with torch.no_grad():
            for images, labels in camera_loader:
                images, labels = images.to(device), labels.to(device)
                camera_correct += (run_model(images).argmax(1) == labels).sum().item()
        record = {'variant': name, 'architecture': architecture, 'seed': seed, 'epochs': epochs,
                  'val_correct': correct, 'val_total': len(task4_val),
                  'val_accuracy': correct / len(task4_val),
                  'camera_correct': camera_correct, 'camera_total': len(camera_ds),
                  'camera_accuracy': camera_correct / len(camera_ds),
                  'seconds': time.perf_counter() - started}
        records.append(record)
        torch.save({key: value.detach().cpu() for key, value in run_model.state_dict().items()},
                   task4_dir / f'{name}_seed{seed}.pt')
        with (task4_dir / 'results.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(records)
        (task4_dir / 'history.json').write_text(json.dumps(history, indent=2))
        print('FINAL:', record)

settings = {'baseline': 'CNN: Task 3d; MobileNet: full fine-tuning without augmentation; final epoch',
            'experiments': experiments, 'seeds': seeds,
            'batch_size': 32, 'optimizer': 'Adam', 'learning_rate': 0.001,
            'mean': mean, 'std': std, 'classes': task4_train.classes,
            'camera_seed': 2026, 'camera_fixture': 'reused unchanged from round 1', 'camera_transform': str(camera_transform),
            'camera_sha256': hashlib.sha256(camera_file.read_bytes()).hexdigest(),
            'torch_version': torch.__version__, 'device': str(device)}
(task4_dir / 'settings.json').write_text(json.dumps(settings, indent=2))


# %%
# Print a table and paired changes, so the report can discuss noise across seeds.
print('| Variant | Seeds | Validation %, mean +/- SD | Camera-like %, mean +/- SD |')
print('| --- | --- | --- | --- |')
for name, *_ in experiments:
    rows = [row for row in records if row['variant'] == name]
    normal = np.array([row['val_accuracy'] * 100 for row in rows])
    camera = np.array([row['camera_accuracy'] * 100 for row in rows])
    normal_sd = normal.std(ddof=1) if len(rows) > 1 else float('nan')
    camera_sd = camera.std(ddof=1) if len(rows) > 1 else float('nan')
    print(f'| {name} | {len(rows)} | {normal.mean():.2f} +/- {normal_sd:.2f} '
          f'| {camera.mean():.2f} +/- {camera_sd:.2f} |')
    if name not in ['baseline', 'mobile_full']:
        for metric in ['val_accuracy', 'camera_accuracy']:
            differences = []
            for row in rows:
                baseline = next(r for r in records
                                if r['variant'] == ('baseline' if row['architecture'] == 'cnn' else 'mobile_full') and r['seed'] == row['seed'])
                differences.append(round(100 * (row[metric] - baseline[metric]), 2))
            print(name, metric, 'paired gains in percentage points:', differences)

# Download the evidence and checkpoints before the Colab runtime expires.
with zipfile.ZipFile('task4_round2.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(task4_dir.iterdir()):
        archive.write(path, arcname=path.name)
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download('task4_round2.zip')


# %%
# OPTIONAL EXPORT: run only after judging results and testing useful combinations.
# Set these two values yourself; this cell does nothing until a model is selected.
chosen_variant = None  # choose a name from the experiment list after judging results
chosen_seed = 0
if chosen_variant is None:
    print('No export yet. Compare results and choose a checkpoint first.')
else:
    if 'google.colab' in sys.modules:
        get_ipython().run_line_magic('pip', 'install -q onnx onnxscript')
    import onnx
    chosen_architecture = next(e[1] for e in experiments if e[0] == chosen_variant)
    if chosen_architecture == 'cnn':
        selected = LeafNetBN()
    else:
        selected = models.mobilenet_v3_small(weights=None)
        selected.classifier[3] = nn.Linear(selected.classifier[3].in_features, 4)
    selected.load_state_dict(torch.load(task4_dir / f'{chosen_variant}_seed{chosen_seed}.pt',
                                        map_location='cpu', weights_only=True))
    selected.eval()
    export_dir = Path('task4_export')
    export_dir.mkdir(exist_ok=True)
    model_path = export_dir / 'leaf_model.onnx'
    torch.onnx.export(selected, (torch.zeros(1, 3, 96, 96),), str(model_path),
                      input_names=['image'], output_names=['logits'], external_data=False)
    selected_info = {'classes': task4_train.classes, 'input_size': 96, 'mean': mean, 'std': std}
    exported = onnx.load(str(model_path))
    exported.metadata_props.add(key='model_info', value=json.dumps(selected_info))
    onnx.checker.check_model(exported)
    onnx.save(exported, str(model_path))
    with Image.open(task4_val.samples[0][0]) as image:
        image.convert('RGB').save(export_dir / 'check_image.jpg', quality=95)
    with Image.open(export_dir / 'check_image.jpg') as image:
        x = plain_transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probabilities = selected(x).softmax(1)[0].tolist()
    (export_dir / 'check_expected.json').write_text(
        json.dumps(dict(zip(task4_train.classes, probabilities)), indent=2))
    (export_dir / 'model_info.json').write_text(json.dumps(selected_info, indent=2))
    with zipfile.ZipFile('task4_export.zip', 'w') as archive:
        for path in sorted(export_dir.iterdir()):
            archive.write(path, arcname=path.name)
    if 'google.colab' in sys.modules:
        files.download('task4_export.zip')
    print('Exported separately. Keep your current app model until this one passes phone tests.')
