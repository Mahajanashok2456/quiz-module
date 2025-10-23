// src/app/components/AdminLoginForm.jsx
"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";

const AdminLoginForm = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const result = await signIn("credentials", {
        redirect: false, // We handle the redirect manually
        email,
        password,
      });

      if (result.error) {
        setError("Invalid email or password. Please try again.");
        setIsLoading(false);
      } else if (result.ok) {
        // On successful login, redirect to the admin dashboard
        router.push("/manage-content-a3f8b1c9"); // <-- Use your actual secret admin route!
      }
    } catch (err) {
      setError("An unexpected error occurred.");
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-sm">
      <h2 className="text-2xl font-bold text-center text-white mb-6">
        Admin Login
      </h2>
      {error && (
        <p className="bg-red-500/30 text-red-400 text-sm text-center p-2 rounded-md mb-4">
          {error}
        </p>
      )}
      <div className="mb-4">
        <label
          className="block text-gray-400 text-sm font-bold mb-2"
          htmlFor="email"
        >
          Email
        </label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="mb-6">
        <label
          className="block text-gray-400 text-sm font-bold mb-2"
          htmlFor="password"
        >
          Password
        </label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition-colors disabled:bg-gray-500"
      >
        {isLoading ? "Signing In..." : "Sign In"}
      </button>
    </form>
  );
};

export default AdminLoginForm;
