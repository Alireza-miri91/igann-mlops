# Resume tomorrow — Azure AKS deploy (runbook)

_Paused 2026-09-10, mid brick #5 stage 2. Follow these steps in order tomorrow._

## Where things stand
- Done: bricks 1–4 (train, MLflow, FastAPI, Docker) + local Kubernetes, all committed.
- Azure so far: **Resource Group `igann-mlops-rg`** and **ACR `igannmlopsacr`** created via Terraform; the **amd64 image is pushed** to ACR (`az acr repository list` shows `igann-mlops`).
- **NOT yet created: the AKS cluster.** That's tomorrow's job.

## Cost / safety
- Safe to shut down the Mac. Only ACR Basic bills (~$0.17/day); no cluster/VM is running.
- Terraform state lives in `terraform/terraform.tfstate` (local, git-ignored). **Do not delete the `terraform/` folder** — it remembers the RG + ACR.

## Tomorrow — do these in order

### 0. Reconnect
```bash
cd "/Users/alirezamiri/Documents/Lebenslauf/tex cv/v1/portfolio/igann-mlops"
az login                 # if it says you're already logged in, fine
az account show          # confirm the right subscription
```
(Start Docker Desktop too, though it's only needed if you rebuild the image.)

### 1. Add the AKS cluster to `terraform/main.tf` (append this)
```hcl
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "igann-mlops-aks"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  dns_prefix          = "igannmlops"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D2s_v7"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
}
```

### 2. Create the cluster (~5–10 min; billing meter starts here)
```bash
cd terraform
terraform apply          # type: yes
```
Wait for "Apply complete! Resources: 2 added."

### 3. Point kubectl at the Azure cluster
```bash
az aks get-credentials --resource-group igann-mlops-rg --name igann-mlops-aks
kubectl config current-context     # should be igann-mlops-aks
kubectl get nodes                  # should show an aks-default-... node
```

### 4. Point the Deployment at the ACR image
In `k8s/deployment.yaml`, change the image line to:
```yaml
          image: igannmlopsacr.azurecr.io/igann-mlops:latest
```

### 5. Deploy and get the public address
```bash
cd ..
kubectl apply -f k8s/
kubectl get pods -w                # wait until both Running, then Ctrl+C
kubectl get service igann          # EXTERNAL-IP is <pending> for ~1 min, then a real IP
```

### 6. Test it live
Open `http://<EXTERNAL-IP>:8000/docs` and run a `/predict`.

### 7. Capture proof (save screenshots to docs/evidence/)
- `terraform apply` output
- Azure Portal: resource group showing AKS + ACR
- Azure Portal: AKS cluster overview (nodes, version, region)
- `kubectl get nodes` (Azure node) and `kubectl get pods,svc` (public IP)
- Browser: live `/docs` + a `/predict` result
- `az acr repository list --name igannmlopsacr -o table`

### 8. TEAR DOWN (important — stops all billing)
```bash
cd terraform
terraform destroy        # type: yes
```
Then confirm it's gone:
```bash
az group show -n igann-mlops-rg    # should error "not found" once destroyed
```
This removes the AKS cluster, ACR, and resource group. After this, spend is $0.

## After that
- Update `docs/BUILD_LOG.md` with the real AKS outputs.
- Remaining bricks: #7 GitHub Actions CI/CD, #8 monitoring (Prometheus + drift).
- End goal (deferred): public repo + LinkedIn post + `bank_experiments` file + CV updates.
